import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from functools import partial

def f(lmbda, v0):
    k = np.sqrt(lmbda)
    kappa = np.sqrt(v0 - lmbda)

    kappa_sin = kappa * np.sin(k/3)
    k_cos = k * np.cos(k/3)

    return np.exp(kappa/3) * (kappa_sin + k_cos)**2 - np.exp(-kappa/3) * (kappa_sin - k_cos)**2

def plot_f(v0, N=1000, eig_vals=[], path=""):
    lmbdas = np.linspace(0, v0, N)

    fig, ax = plt.subplots()
    plt.plot(lmbdas, f(lmbdas, v0), color='k', label=r"$f(\lambda)$")
    y_lims = ax.get_ylim()
    
    # Plot eigenvalues from numerical scheme
    if len(eig_vals):
        plt.vlines(eig_vals, *y_lims, linestyles="--", color="k", label="Numerical\neigenvalues")

        # Text
        text_i = []
        for i in range(len(eig_vals)):
            if i < len(eig_vals) - 1 and abs(eig_vals[i] - eig_vals[i+1]) < 10:
                text_i.append(str(i+1))
            else:
                text_i.append(str(i+1))
                plt.text(eig_vals[i]+10, y_lims[1] * (1-3e-2), f"n={','.join(text_i)}")
                text_i = []


    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$f(\lambda)$")
    plt.legend(loc="center right")
    # plt.title("")
    plt.show()
    if len(path) > 0:
        fig.savefig(path)

def find_root(func, init_guess):
    return root_scalar(func, x0=init_guess).root

def find_roots(v0, eig_vals):
    roots = []
    for i, eig_val in enumerate(eig_vals):
        init_guess = eig_val + 3 * (-1)**((i+1)%2)
        roots.append(find_root(partial(f, v0=v0), init_guess=init_guess))

    for i in range(len(roots)):
        print(f'Lmbda={eig_vals[i]} -- Root={roots[i]} -- Diff={abs(eig_vals[i]-roots[i])}')

if __name__ == '__main__':
    plot_f(1e3)