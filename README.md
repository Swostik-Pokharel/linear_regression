# Linear Regression with Gradient Descent

A from-scratch implementation of linear regression using gradient descent optimization with real-time visualization.

## Overview

This project implements simple linear regression without high-level ML libraries to understand the core mechanics of gradient descent and optimization. Features a live animation showing the model converging to the optimal solution.

## Features

- **Simple linear regression** (y = mx + b) for 2-variable datasets
- **Batch gradient descent** implementation from scratch
- **Real-time animation** showing convergence during training
- **Live metrics display** (epoch, parameters, loss) during visualization
- **Custom dataset generator** with adjustable scatter levels (1-5)

## How It Works

The algorithm minimizes mean squared error by iteratively adjusting slope (m) and intercept (b):
- Calculates gradients of the loss function with respect to m and b
- Updates parameters using learning rate
- Visualizes the regression line at each epoch

## Visualization

Uses matplotlib's animation module to show:
- Static scatter plot of data points
- Animated regression line adjusting in real-time
- Current epoch, parameters (m, b), and loss value
- Configurable FPS for animation speed

## Dependencies

```
numpy
matplotlib
pandas
```

## Usage

1. Generate synthetic data with `data/generator.py` (optional)
2. Run `main.py` and provide CSV path
3. Watch the live gradient descent animation

## Status

✅ Complete
