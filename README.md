# Model-Fix: Automated ML Model Diagnostics and Fixes

## Overview

Model-Fix is an automated machine learning system designed to enhance your ML workflow by diagnosing common model training issues and applying targeted fixes. This internship-level project demonstrates practical ML engineering skills including model training, diagnostics, automated hyperparameter tuning, and visualization.

The system focuses on a simple Multi-Layer Perceptron (MLP) trained on the MNIST dataset, automatically detecting problems like overfitting and underfitting, then applying appropriate fixes without manual intervention.

## Features

- **Automated Diagnostics**: Analyzes training history to detect overfitting, underfitting, or good fit
- **Intelligent Fixes**: Applies targeted solutions based on diagnosed problems (dropout, model size adjustments, etc.)
- **Visualization**: Plots training curves and before/after comparisons
- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, models, training, and utilities
- **Extensible Design**: Easy to add new diagnostics, fixes, and model architectures

## Project Structure

```
Model-fix/
├── auto.py                 # Main automation logic for running experiments with fixes
├── test.py                 # Demo script showing the full pipeline
├── README.md              # This file
├── data/
│   ├── dataloader.py      # MNIST data loading and preprocessing
│   └── MNIST/             # Downloaded MNIST dataset
├── models/
│   └── mlp.py             # Simple MLP model definition
├── training/
│   └── train.py           # Training loop implementation
└── utility/
    ├── diagnosis.py       # Problem detection algorithms
    ├── explain.py         # Human-readable explanations
    ├── plot.py            # Visualization functions
    └── suggestion.py      # Fix suggestions
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Model-fix
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

## Usage

### Basic Training and Diagnostics

Run the demo script to see the full pipeline:

```bash
python test.py
```

This will:
1. Train a baseline MLP model on MNIST
2. Diagnose any training issues
3. Apply automated fixes
4. Visualize the results

### Manual Usage

```python
from data.dataloader import get_dataloaders
from models.mlp import SimpleMLP
from training.train import train_model
from utility.diagnosis import diagnose
from utility.explain import explain
from utility.suggestion import suggest
from auto import run_experiment
import torch

# Load data
train_loader, val_loader = get_dataloaders()

# Train baseline model
model = SimpleMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20)

# Diagnose issues
problem = diagnose(history)
print(f"Diagnosis: {problem}")
print(explain(problem))

# Get suggestions
suggestions = suggest(problem)
for suggestion in suggestions:
    print(f"- {suggestion}")

# Apply automated fix
new_history, new_config = run_experiment(train_loader, val_loader, problem, {
    "dropout": 0.0,
    "hidden_size": 256,
    "lr": 0.001,
    "epochs": 20,
    "optimizer": torch.optim.Adam,
    "criterion": torch.nn.CrossEntropyLoss()
})
```

## Learning Outcomes

This project demonstrates:
- PyTorch model implementation and training
- Data preprocessing and loading
- Automated hyperparameter tuning
- Model diagnostics and debugging
- Code organization and modularity
- Visualization for ML experiments
- Problem-solving in machine learning workflows

## Future Enhancements

- Support for additional model architectures (CNNs, transformers)
- More sophisticated diagnostics (gradient analysis, activation patterns)
- Integration with MLflow for experiment tracking
- Web interface for interactive model fixing
- Support for custom datasets and problems

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

