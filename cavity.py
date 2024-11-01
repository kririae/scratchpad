#!/usr/bin/env python

import pickle
from cProfile import label

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

y = np.array([
    1.0000,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000,
])
ref_Re1000 = np.array([
    1.00000,
    0.65928,
    0.57492,
    0.51117,
    0.46604,
    0.33304,
    0.18719,
    0.05702,
    -0.06080,
    -0.10648,
    -0.27805,
    -0.38289,
    -0.29730,
    -0.22220,
    -0.20196,
    -0.18109,
    0.00000,
])


def u_Re1000():
    with open("u_MIGKS_Re1000.pickle", "rb") as f:
        u_MIGKS = pickle.load(f)
    N = 500
    dx = 1.0 / N
    plt.plot(
        np.linspace(0 + 0.5 * dx, 1.0 - 0.5 * dx, N),
        u_MIGKS[N // 2 + 1, 1:-1, 0] / 0.18,
        "g-",
        label="M-IGKS",
    )
    plt.plot(y, ref_Re1000, "rs")
    plt.xlabel("y")
    plt.ylabel("$u_x$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    u_Re1000()
