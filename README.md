# hierarchical-lab

A hands-on pathway for one idea worth internalizing:

> **When you know structure among your outputs, bake it into the architecture *and* the objective.**

CIFAR-100's 100 fine classes nest under 20 coarse superclasses — a real taxonomy to exploit.

> ## 🙏 Credits — this is a learning project built on someone else's work
>
> This repo exists because of **[Ugenteraan/Deep_Hierarchical_Classification](https://github.com/Ugenteraan/Deep_Hierarchical_Classification)**
> by **[Ugenteraan Manogaran](https://github.com/Ugenteraan)** (MIT, © 2021). I read that repo closely and
> built this as a study pathway around its ideas — the **coarse-to-fine classifier head** and the
> **hierarchical loss formulation** here follow its approach directly. Please go star the original.
>
> That project is an unofficial implementation of **"Deep Hierarchical Classification for Category
> Prediction in E-commerce System"**, Wan et al. 2020 — [arXiv:2005.06692](https://arxiv.org/abs/2005.06692).
>
> Where this repo critiques the original (see *"The bug that's the best lesson"* below), it's offered in
> good faith as a technical observation about one implementation detail — not as a knock on a project that
> was generous enough to be public and readable in the first place. Being able to read real code and learn
> from it is the whole point. See [`NOTICE`](NOTICE) for full attribution details.

Runs on Apple-Silicon **MPS** in seconds/epoch with a small CNN, so you can iterate and *watch* what changes.

## The two ideas we keep

1. **Coarse-to-fine conditioning (architecture).** One shared backbone → a coarse head (20 logits) whose output
   is *concatenated into* the fine head (`cat(20, 100) → 100`). The fine decision is conditioned on the coarse
   one. See `models.py: CoarseToFineModel` — faithful to the repo's `model/resnet50.py`.

2. **Put the taxonomy in the loss.**
   - **Layer loss** (`lloss`): cross-entropy at every level, summed. Deep supervision. `losses.layer_loss`.
   - **Dependency loss**: penalize a fine prediction that isn't a child of the predicted coarse class.

## The bug that's the best lesson

The repo's dependency loss is:

```
D_l    = 1 if argmax(fine) is NOT a child of argmax(coarse) else 0
l_prev = 0 if argmax(coarse)==coarse_true else 1
l_curr = 0 if argmax(fine)  ==fine_true   else 1
dloss  = Σ  p^(D_l·l_prev) · p^(D_l·l_curr) − 1        # p = 3
```

Every term comes from `argmax` / `==` / `where` — all **non-differentiable**. So `dloss.requires_grad` is
`False`; it can't even `.backward()`. Their headline "respect the hierarchy" mechanism contributes **zero
gradient** — it's a monitoring number dressed as a loss. Stage 4 proves this, then replaces it with a
differentiable version:

```
SoftHierarchyLoss = −log( Σ fine-class probability mass inside the TRUE superclass )
                  = NLL of the true parent under the fine distribution marginalized up the tree
```

The fix is to stop asking *"which class won?"* (a step function — flat almost everywhere, so zero gradient)
and start asking *"how much probability landed in the right block?"* (smooth in every logit).

The linear algebra: let `M` be the `(20 × 100)` membership matrix with `M[c,f] = 1` iff `f` is a child of `c`.
Every column holds exactly one 1, so `M` is the one-hot encoding of parenthood — and `probs @ Mᵀ` is precisely
the vector of parent marginals. **Marginalizing up a tree is a matrix product**; the taxonomy stops being
control flow and becomes a linear operator, which is why it's differentiable at all.

In code it's evaluated in log-space (`log_softmax` → `masked_fill(-inf)` → `logsumexp`) rather than literally
forming `probs @ Mᵀ`. Same quantity, but exact in the tail: once the mass in a parent drops below ~1e-45 it
flushes to 0 in float32 and the naive form returns `log(0) = -inf`, usually patched with an `eps` that silently
caps the loss and biases the gradient. There's a test pinning this exact case.

See `losses.SoftHierarchyLoss` and the reusable `structured_loss.SoftHierarchyLoss`.

## Results — measured, not claimed

Every number below is from an actual run on this code (Apple M-series MPS, seed 0, identical
initialization and data order across variants). Reproduce with the commands in *Run it*.

**Stage 4 — 4 losses, 6 epochs, same seed.** `violation %` = how often the fine prediction's parent
disagrees with the coarse head's prediction.

| objective | fine acc | violation % |
|---|---|---|
| `lloss` only (baseline) | 43.25% | 26.39% |
| `lloss` + **their dloss** | **43.25%** | **26.39%** ← *identical to baseline, to the last decimal* |
| `lloss` + soft loss (fine → TRUE parent) | 40.95% | 28.07% ← *did not help* |
| `lloss` + **head-agreement KL(m‖c)** | 42.88% | **20.52%** ← *the fix* |

**Stages 1–3 — 8 epochs.**

| stage | fine acc | coarse acc | violation % |
|---|---|---|---|
| 1 · flat baseline | 49.46% | 63.31% | 36.69% \* |
| 2 · two heads + layer loss | 50.79% | 62.86% | 23.08% |
| 3 · coarse-to-fine conditioning | 48.16% | 60.42% | 24.17% |

\* Stage 1 has one head, so there is no second opinion to contradict; its "violation" is measured against
the *true* parent and is just `100 − coarse acc`. **It is not comparable to stages 2–3**, which measure
disagreement *between the two heads*. Different question, different number.

Two honest negatives worth stating plainly: **Stage 3's conditioning did not beat Stage 2** here
(-2.6 pts fine accuracy, slightly worse violation) — the upstream README's headline gain does not
reproduce at this scale, though this is a small CNN at 8 epochs, not their ResNet50 at 100. And my
first attempt at fixing the dependency loss failed, which turned out to be the most useful thing that happened:

## The journey — how the fix was actually found

1. **Read the upstream `dloss` and noticed it's built from `argmax`/`==`/`where`.** Predicted it carries no
   gradient. *Evidence:* `requires_grad=False`, `grad_fn=None`, and it raises on `.backward()`.
2. **Proved it behaviorally, not just theoretically.** Trained baseline vs baseline+dloss from the same seed.
   The trajectories are **bit-for-bit identical at every epoch** — while the term itself reports a value of
   ~790. A loss can be large, look active in your logs, and be doing *nothing*.
3. **Wrote `SoftHierarchyLoss`** (mass in the true parent, via marginalization). Differentiable, gradient
   reaches the backbone, mathematically sound — and it **failed to improve the violation rate** (28.07% vs
   26.39% baseline).
4. **Diagnosed why, and this is the real lesson.** The loss ties the fine head to the **true** parent label.
   The metric compares the fine head against the **coarse head's prediction**. The truth never appears in the
   metric. So the loss can be fully satisfied while the two heads still contradict each other — *I had
   optimized a different consistency than the one I was measuring.*
5. **Wrote `HeadAgreementLoss` = KL(m ‖ c)** — marginalize the fine distribution to the parents and make it
   agree with the coarse head's own distribution. Now the measured quantity is *in* the objective.
   **Violation 26.39% → 20.52%** at no meaningful accuracy cost.

The transferable lesson is not "use KL." It's that **a regularizer can be differentiable, correct, and
well-motivated and still do nothing for your metric, because it doesn't optimize the thing you're measuring.**
Being able to see that required a controlled A/B and a metric separate from accuracy. Step 3 was not wasted
work — it was the experiment that located the real problem.

## Other things I changed (and why)

- **MPS.** Repo is `cuda`-or-`cpu` only; on a Mac it silently runs on CPU. `device.pick_device` is MPS-first.
- **Data pipeline.** Repo dumps ~60k PNGs + a CSV then re-reads with `cv2` (and has a dead `cv2.resize`).
  Replaced by `torchvision.datasets.CIFAR100` + the canonical fine→coarse map (`hierarchy.py`, ~1 file).
- **Visualization.** Repo = three seaborn line charts. Here: taxonomy tree, block-structured confusion matrix
  (errors inside vs. across superclass blocks), violation-rate comparison, and a PCA embedding-evolution GIF.
- **Controlled comparisons** instead of the repo's "+10%, but maybe it's the augmentation" hedge.

## The pathway

| stage | script | idea | artifact |
|------|--------|------|----------|
| 0 | `s0_tree.py` | see the taxonomy | `runs/hierarchy_tree.png` |
| 1 | `s1_flat_baseline.py` | flat 100-way; measure implied violations | `runs/s1_confusion.png` |
| 2 | `s2_two_head_lloss.py` | coarse+fine heads, layer loss | `runs/s2_*` |
| 3 | `s3_coarse_to_fine.py` | concat coarse logits into fine head | `runs/s3_*` |
| 4 | `s4_dependency_loss.py` | prove dloss is dead → find the loss that actually works | `runs/violation_comparison.png` |
| 5 | `s5_representation_evolution.py` | watch superclasses cluster | `runs/embedding_evolution.gif` |
| 6 | `structured_loss.py` | reusable, dataset-agnostic toolkit | — |

## Run it

```bash
uv sync
uv run python s0_tree.py                       # instant, no download
uv run python s1_flat_baseline.py --epochs 8
uv run python s4_dependency_loss.py --epochs 6 # the money shot
uv run python s5_representation_evolution.py --epochs 12
```

All scripts: `--epochs --lr --batch-size --seed --device`. Artifacts land in `runs/` (gitignored).

> **Note on the dataset:** on macOS, torchvision's downloader can hit `CERTIFICATE_VERIFY_FAILED`. If so,
> pre-fetch once with the system trust store:
> `curl -fsSL -o data/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`
> then run normally (torchvision extracts the local tarball).

## Where this transfers

Any task with known output structure: LLM classification/routing over a label taxonomy, cascaded coarse-to-fine
decoders, product/document categorization, ICD/ontology codes. The reusable pattern lives in `structured_loss.py`
— point it at any `child_to_parent` map. The meta-lesson: **enforce structure by marginalizing probabilities,
never by argmax** — argmax has no gradient.
