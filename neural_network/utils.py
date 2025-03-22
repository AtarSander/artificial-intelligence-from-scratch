from PIL import Image as im
import time
import matplotlib.pyplot as plt
import numpy as np


def view_image(image: np.ndarray) -> None:
    data = im.fromarray(image)
    data.save("test.png")


def timer(func):
    def wrapper(*args, **kwargs):
        print("Starting the training")
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print("Training finished")
        exec_time = round(end - start, 2)
        return ret, f"{exec_time}s"

    return wrapper


def plot_loss(costs: list[list[float]]) -> None:
    x_values_modified = [x for x in range(len(costs[0]))]
    for ind, cost in enumerate(costs):
        plt.plot(
            x_values_modified,
            cost,
            marker="",
            linestyle="-",
            label=f"Model number: {ind}",
        )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average loss of models during training")
    plt.legend()
    plt.show()
