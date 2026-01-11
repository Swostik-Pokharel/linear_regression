import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loss_function(
    m, b, datapoints
):  # we dont use this function but its helpful for representation
    error = 0
    for i in range(len(datapoints)):
        x = datapoints.iloc[i].x
        y = datapoints.iloc[i].y
        error += (y - (m * x + b)) ** 2
    return error / float(len(datapoints))


def gradient_descent(datapoints, m_curr, b_curr, L, n=None):
    gradient_m = 0
    gradient_b = 0
    if (
        n is None
    ):  # this is for later when i use stocastic gradient descent i can set my own n for each iteration
        n = len(datapoints)
    for i in range(n):
        x = datapoints.iloc[i].x
        y = datapoints.iloc[i].y

        # we take the sum of the partial derivatives of total error in respect to each parameter
        gradient_m += -(2 / n) * x * (y - (m_curr * x + b_curr))
        gradient_b += -(2 / n) * (y - (m_curr * x + b_curr))
    m, b = m_curr - gradient_m * L, b_curr - gradient_b * L
    return m, b


def main():
    filepath = input("Enter the filpath for the csv:  ")
    data = pd.read_csv(filepath)
    print(
        f"Data from file {filepath} loaded\n preview:\n {data.head()} \n row,col count ={data.shape}"
    )
    # where the required line is of eq y=mx+b
    m = 0
    b = 0
    L = 0.0001  # learning rate
    epochs = 10000

    for n in range(epochs):
        if n % 100 == 0:
            print(f"{n}th iteration")
        m, b = gradient_descent(data, m, b, L)
    print(f"the line of best fit is y ={m}x+{b}")
    plt.scatter(data.x, data.y, color="black")
    plt.plot(list(range(0, 150)), [m * x + b for x in range(0, 150)], color="blue")
    plt.show()


if __name__ == "__main__":
    main()
