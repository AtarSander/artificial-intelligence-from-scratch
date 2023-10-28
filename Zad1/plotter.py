from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def plotter_1D(func, domain, grad_x=None, grad_y=None, num_points=400,
               title=None, x_label='x', y_label='y', legend_label=None,
               steps=False):

    x = np.linspace(domain[0], domain[1], num_points)
    y = func(x)
    plt.plot(x, y)

    if steps:
        for i in range(1, len(grad_x)):
            plt.annotate("", xy=(grad_x[i], grad_y[i]),
                         xytext=(grad_x[i-1], grad_y[i-1]),
                         arrowprops={'arrowstyle': "->",
                                     'color': 'r', 'lw': 1},
                         va='center', ha='center')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if legend_label:
        plt.legend(legend_label)
    plt.grid(True)
    plt.show()


def plotter_2D(func, domain=[(-10, 10), (-10, 10)], grad_x=None, num_points=400,
               title=None, x1_label='x1', x2_label='x2', legend_label=None, steps=False):
    #TODO: add two options of showing arrows (every arrow vs every so steps)
    x1 = np.linspace(domain[0][0], domain[0][1], num_points)
    x2 = np.linspace(domain[1][0], domain[1][1], num_points)
    X1, X2 = np.meshgrid(x1, x2)

    Z = func((X1, X2))
    plt.contourf(X1, X2, Z, cmap='viridis')
    plt.colorbar()
    if steps:
        for i in range(1, len(grad_x)):
            plt.annotate("", xy=grad_x[i], xytext=grad_x[i-1],
                         arrowprops={'arrowstyle': "->", 'color': 'r', 'lw': 1},
                         va='center', ha='center')

    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    if title:
        plt.title(title)
    if legend_label:
        plt.legend(legend_label)

    plt.show()


def plotter_3D(func, domain, grad_x=None, num_points=100, title=None,
               view=(15, -30), x1_label='x1', x2_label='x2',
               x3_label='f(x1, x2)', legend_label=None, steps=False):
    x1 = np.linspace(domain[0][0], domain[0][1], num_points)
    x2 = np.linspace(domain[1][0], domain[1][1], num_points)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func((X1, X2))
    # fig = plt.figure(figsize=[12, 8])
    ax = plt.axes(projection="3d")
    ax.plot_surface(X1, X2, Z, cmap="plasma")
    ax.view_init(*view)

    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    # plt.zlabel(x3_label)
    if title:
        plt.title(title)
    if legend_label:
        plt.legend(legend_label)
    plt.show()
