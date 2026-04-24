# Self-Pruning Neural Network

## Overview

This project implements a **Self-Pruning Neural Network** in PyTorch that learns to remove unnecessary weights *during training* instead of relying on post-training pruning techniques.

The model introduces **learnable gate parameters** for each weight in fully connected layers. These gates control whether a weight should remain active or be gradually suppressed, allowing the network to optimize both **accuracy** and **model sparsity** at the same time.

This project was developed as an **AI Engineer Case Study** and demonstrates practical neural network compression using differentiable pruning.

---

## Key Idea

Instead of pruning weights after training, each weight is assigned a learnable gate:

[
\text{gates} = \sigma(\text{gate_scores}) \in (0,1)
]

[
\text{pruned_weights} = W \odot \text{gates}
]

[
\text{output} = X \cdot \text{pruned_weights}
]

Where:

* `gate_scores` are trainable parameters
* `sigmoid()` converts them into soft gates
* low gate values gradually suppress unnecessary weights
* L1 regularization on gates encourages sparsity

This allows the model to learn **which weights are important and which can be removed**.

---

## Features

* Custom `PrunableLinear` layer
* Learnable differentiable pruning mechanism
* CIFAR-10 image classification
* ConvNet feature extractor + prunable dense layers
* Automatic sparsity learning using L1 gate regularization
* Gradient flow verification for pruning parameters
* Lambda (λ) sweep for sparsity vs accuracy tradeoff
* Performance visualization and summary tables

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pandas
* Jupyter Notebook

---

## Dataset

This project uses the **CIFAR-10** dataset from `torchvision.datasets`.

CIFAR-10 contains:

* 60,000 color images
* 10 classes
* 32x32 image size

Classes include:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

---

## Model Architecture

### Feature Extractor

A CNN backbone is used for representation learning:

* Conv → BatchNorm → ReLU → MaxPool
* Multiple stacked convolution blocks

### Classification Head

Fully connected layers are replaced with:

* `PrunableLinear`

This is where self-pruning happens.

---

## Training Strategy

The total loss is:

[
\mathcal{L}*{total} = \mathcal{L}*{classification} + \lambda \cdot \mathcal{L}_{sparsity}
]

Where:

* classification loss = CrossEntropyLoss
* sparsity loss = L1 penalty on gate values
* λ controls pruning strength

### Lambda Sweep

The project trains models with:

```python
λ ∈ {0, 1e-5, 1e-4, 5e-4, 1e-3}
```

This helps analyze the tradeoff between:

* model accuracy
* sparsity level

---

## Results

The notebook includes:

* test accuracy comparison
* sparsity percentage analysis
* validation curves
* training performance plots
* summary result tables

This shows how increasing λ improves sparsity while affecting accuracy.

---

## Project Structure

```text
self_pruning_neural_network.ipynb
README.md
/data
```

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/self-pruning-neural-network.git
cd self-pruning-neural-network
```

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib numpy pandas jupyter
```

### 3. Run Notebook

```bash
jupyter notebook
```

Open:

```text
self_pruning_neural_network.ipynb
```

---

## Learning Outcome

This project demonstrates:

* neural network compression
* differentiable pruning
* sparsity regularization
* model efficiency optimization
* practical PyTorch custom layer design

It is highly relevant for:

* AI Engineer roles
* Deep Learning projects
* Model Compression research
* Edge AI deployment

---

## Future Improvements

Possible extensions:

* Structured pruning
* Channel pruning for CNN layers
* Hard threshold pruning after training
* ONNX / TensorRT deployment
* Pruning-aware quantization
* Comparison with Lottery Ticket Hypothesis

---

## Author

Developed as part of an AI Engineer Case Study project focused on efficient deep learning systems and neural network optimization.

---

## License

This project is for academic, research, and portfolio purposes.
