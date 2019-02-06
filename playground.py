import cmath
import matplotlib.pyplot as plt


def gauss_neighborhood(d, t):
    return pow(cmath.e, -(d/2*t))


def mexican_hat_neighborhood(d, t):
    return (1 - (d/t)) * pow(cmath.e, -(d/2*t))


if __name__ == '__main__':
    d = 5
    results = []
    for t in range(1, 10):
        results.append(mexican_hat_neighborhood(d, t))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(results)
    plt.show()
