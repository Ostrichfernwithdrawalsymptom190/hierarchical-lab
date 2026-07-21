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
SoftHierarchyLoss = −log( Σ fine_softmax mass landing inside the TRUE superclass )
                  = NLL of the true parent under the fine distribution marginalized up the tree
```

Marginalize the fine softmax up to the parent (`probs @ Mᵀ`) and take the NLL of the true coarse label.
Gradient flows into the fine logits; the model actually learns to keep its mass in the right superclass.
See `losses.SoftHierarchyLoss` and the reusable `structured_loss.SoftHierarchyLoss`.

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
| 4 | `s4_dependency_loss.py` | prove dloss is dead → soft fix | `runs/violation_comparison.png` |
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
