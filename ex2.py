import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random


def dot_product(w, x):
    result = 0
    for x_i, w_i in zip(w, x):
        result += x_i * w_i
    return result


def multiply_vector(k, x):
    result = []
    for x_i in x:
        result.append(k * x_i)
    return result


def vector_add(x, y):
    result = []
    for x_i, y_i in zip(x, y):
        result.append(x_i + y_i)
    return result


def vector_subtr(x, y):
    result = []
    for x_i, y_i in zip(x, y):
        result.append(x_i - y_i)
    return result


def step_function(x):
    if x >= 0:
        return 1
    return 0


def plot_boundary(weights, df, iter):
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(-4, 6, 1000)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 3])
    plt.plot(x, ((-weights[0] * x) - weights[2]) / weights[1], c='r')
    plt.xlim(-4, 6)
    plt.ylim(-5, 5)
    plt.title("iteration #{:d}, weights=[{:.2f}, {:.2f}, {:.2f}]".format(iter, weights[0], weights[1], weights[2]))
    plt.savefig("ex2_outputs/{}".format(iter))
    #plt.show()
    plt.close(fig)


if __name__ == '__main__':
    df = pd.read_csv("points.csv")
    df = df.sample(frac=1)
    weights = [1, -1, 2]
    epsilon = 0.05
    plot_boundary(weights, df, 0)
    iter = 1

    for i in range(4):
        for index, row in df.iterrows():
            if row['class'] != step_function(dot_product(weights, row[:3])):
                #weights = vector_subtr(weights, multiply_vector(epsilon, vector_subtr(weights, row[:3])))

                weights = vector_add(
                    weights,
                    multiply_vector(epsilon * (row['class'] - step_function(dot_product(weights, row[:3]))), row[:3]))
            plot_boundary(weights, df, iter)
            iter += 1

