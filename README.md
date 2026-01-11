# Linear Regression with Gradient Descent

A from-scratch implementation of linear regression using gradient descent optimization.

## Overview

This project implements linear regression without high-level ML libraries to understand the core mechanics of gradient descent and optimization.

## Features

### Phase 1: Simple Linear Regression
- 2-variable linear regression (y = mx + b)
- Batch gradient descent implementation
- Matplotlib visualization of regression line

### Phase 2: Multivariate Regression
- Support for multiple input features
- Automatic switching between batch GD and stochastic GD based on dataset size
- Feature normalization
- L1 (Lasso) and L2 (Ridge) regularization options

### Phase 3: Training Visualization
- Capture gradient descent iterations as frames
- Generate video showing how the model converges to the optimal solution
- Visual comparison of different regularization methods and their effect on convergence

## Algorithm Switching

- Dataset < 10,000 samples → Batch Gradient Descent
- Dataset ≥ 10,000 samples → Stochastic Gradient Descent

## Dependencies

```
numpy
matplotlib
pandas
opencv-python
```

## Status

Phase 1: Complete | Phase 2: In Progress | Phase 3: Planned
