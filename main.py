import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from pathlib import Path


def loss_function(m, b, datapoints):
    """Calculate mean squared error (for reference, not used in training)"""
    error = 0
    for i in range(len(datapoints)):
        x = datapoints.iloc[i].x
        y = datapoints.iloc[i].y
        error += (y - (m * x + b)) ** 2
    return error / len(datapoints)


def gradient_descent(datapoints, m_curr, b_curr, learning_rate, n=None):
    """Perform one step of gradient descent"""
    gradient_m = 0
    gradient_b = 0

    if n is None:
        n = len(datapoints)

    for i in range(n):
        x = datapoints.iloc[i].x
        y = datapoints.iloc[i].y
        # Partial derivatives of MSE with respect to m and b
        gradient_m += -(2 / n) * x * (y - (m_curr * x + b_curr))
        gradient_b += -(2 / n) * (y - (m_curr * x + b_curr))

    m = m_curr - gradient_m * learning_rate
    b = b_curr - gradient_b * learning_rate

    return m, b


def train_and_animate(data, learning_rate=0.0001, epochs=100, fps=12):
    """Train model and show live animation of gradient descent"""

    # Initialize parameters
    m_values = [0]
    b_values = [0]

    # Train and store parameters at each epoch
    print("Training model...")
    for epoch in range(epochs):
        m, b = gradient_descent(data, m_values[-1], b_values[-1], learning_rate)
        m_values.append(m)
        b_values.append(b)

        if epoch % 10 == 0:
            loss = loss_function(m, b, data)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

    print(f"\nFinal line of best fit: y = {m_values[-1]:.4f}x + {b_values[-1]:.4f}")

    # Set up animation
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data points (static)
    ax.scatter(data.x, data.y, color="black", s=30, alpha=0.6, label="Data points")

    # Set up line for animation
    x_range = np.linspace(data.x.min() - 10, data.x.max() + 10, 100)
    (line,) = ax.plot([], [], "b-", linewidth=2, label="Best fit line")

    # Configure plot
    ax.set_xlim(data.x.min() - 10, data.x.max() + 10)
    ax.set_ylim(data.y.min() - 10, data.y.max() + 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Text for displaying epoch and parameters
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    def init():
        """Initialize animation"""
        line.set_data([], [])
        text.set_text("")
        return line, text

    def update(frame):
        """Update animation frame"""
        m = m_values[frame]
        b = b_values[frame]

        # Update line
        y_range = m * x_range + b
        line.set_data(x_range, y_range)

        # Update text
        loss = loss_function(m, b, data)
        text.set_text(f"Epoch: {frame}\nm = {m:.4f}\nb = {b:.4f}\nLoss = {loss:.4f}")

        return line, text

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(m_values),
        init_func=init,
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )

    plt.title("Gradient Descent Animation")
    plt.show()


def main():
    # Get input
    filepath = input("Enter the filepath for the CSV: ")

    # Load data
    data = pd.read_csv(filepath)
    print(f"\nData from '{filepath}' loaded")
    print(f"Preview:\n{data.head()}")
    print(f"Shape: {data.shape}")

    # Training parameters
    learning_rate = 0.0001
    epochs = 24
    fps = 4

    # Train and animate
    train_and_animate(data, learning_rate, epochs, fps)


if __name__ == "__main__":
    main()
