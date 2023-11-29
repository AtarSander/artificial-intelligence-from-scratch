from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def plotter_1D(func, domain, grad_x=None, grad_y=None, num_points=400,
               title=None, x_label='x', y_label='y', legend_label=None,
               steps=False, every=False):
    """
    Creates a 1D plot for given function with/without grad steps

    :param func: function to plot
    :type: funct

    :param domain: domain of the plot
    :type: tuple(int)

    :param grad_x: x coordinate gradient steps
    :type: list

    :param grad_y: y coordinate gradiet steps
    :type: list

    :param num points: how many points in a plot space
    :type: int

    :param title: plot title
    :type: str

    :param x_label: x axis title
    :type: str

    :param y_label: y axis title
    :type: str

    :param legend_label: function legend label
    :type: str

    :param steps: show/not show gradient steps
    :type: bool

    :param every: show every iter grad step/show every 10 iter grad step
    :type: bool
    """

    x = np.linspace(domain[0], domain[1], num_points)
    y = func(x)
    plt.plot(x, y, label=legend_label)

    if steps:
        for i in range(1, len(grad_x)):
            if i % 10 == 0 or every:
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
        plt.legend()
    plt.grid(True)
    plt.show()


def plotter_2D(func, domain=[(-10, 10), (-10, 10)], grad_x=None, num_points=400,
               title=None, x1_label='x1', x2_label='x2', legend_label=None,
               steps=False, every=False):
    """
    Creates a 2D plot for given function with/without grad steps

    :param func: function to plot
    :type: funct

    :param domain: domain of the plot
    :type: list(tuple(int))

    :param grad_x: x coordinate gradient steps
    :type: np.array

    :param num points: how many points in a plot space
    :type: int

    :param title: plot title
    :type: str

    :param x1_label: x axis title
    :type: str

    :param x2_label: y axis title
    :type: str

    :param legend_label: function legend label
    :type: str

    :param steps: show/not show gradient steps
    :type: bool

    :param every: show every iter grad step/show every 10 iter grad step
    :type: bool
    """
    x1 = np.linspace(domain[0][0], domain[0][1], num_points)
    x2 = np.linspace(domain[1][0], domain[1][1], num_points)
    X1, X2 = np.meshgrid(x1, x2)

    Z = func((X1, X2))
    plt.contourf(X1, X2, Z, cmap='viridis')
    plt.colorbar()
    if steps:
        for i in range(1, len(grad_x)):
            if i % 10 == 1 or every:
                plt.annotate("", xy=grad_x[i], xytext=grad_x[i-1],
                             arrowprops={'arrowstyle': "->", 'color': 'r', 'lw': 1},
                             va='center', ha='center')

    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    if title:
        plt.title(title)
    plt.show()


def plotter_3D(func, domain, grad_x=None, grad_y=None, num_points=100, title=None,
               view=(15, -30), x1_label='x1', x2_label='x2', x3_label='f(x1, x2)',
               steps=False, every=False):
    """
    Creates a 3D plot for given function with/without grad steps

    :param func: function to plot
    :type: funct

    :param domain: domain of the plot
    :type: list(tuple(int))

    :param grad_x: x coordinate gradient steps
    :type: np.array

    :param grad_y: y coordinate gradient steps
    :type: np.array

    :param num points: how many points in a plot space
    :type: int

    :param title: plot title
    :type: str

    :param view: how the 3D plot is positioned
    :type: tuple(int)

    :param x1_label: x axis title
    :type: str

    :param x2_label: y axis title
    :type: str

    :param x3_label: z axis title
    :type: str

    :param legend_label: function legend label
    :type: str

    :param steps: show/not show gradient steps
    :type: bool

    :param every: show every iter grad step/show every 10 iter grad step
    :type: bool
    """
    x1 = np.linspace(domain[0][0], domain[0][1], num_points)
    x2 = np.linspace(domain[1][0], domain[1][1], num_points)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func((X1, X2))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X1, X2, Z, cmap="plasma", alpha=0.5)
    ax.view_init(*view)

    if steps:
        grad = np.column_stack((grad_x, grad_y))
        for i in range(1, len(grad_x)):
            if i % 10 == 1 or every:
                ax.quiver(*grad[i-1], *grad[i], color='g', label='Arrow', zorder=10)

    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    ax.set_zlabel(x3_label)
    if title:
        plt.title(title)
    plt.show()
