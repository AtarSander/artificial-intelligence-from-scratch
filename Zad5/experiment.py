from model import Model
import matplotlib.pyplot as plt


def experiment(list_of_layers, learning_rates, data, batch_sizes, num_epochs):
    costs = []
    results = {}
    train_accuracy_column = []
    dev_accuracy_column = []
    max_accuracy = 0
    X_train, Y_train, X_dev, Y_dev = data
    for i in range(len(learning_rates)):
        new_model = Model(list_of_layers[i])
        cost = new_model.train(
            X_train,
            Y_train,
            learning_rates[i],
            num_epochs[i],
            batch_sizes[i],
            1,
            False,
        )
        costs.append(cost)
        acc_dev = new_model.predict(X_dev, Y_dev)
        acc_trn = new_model.predict(X_train, Y_train)
        train_accuracy_column.append(acc_trn)
        dev_accuracy_column.append(acc_dev)
        if acc_dev > max_accuracy:
            max_accuracy = acc_dev
            best_model = new_model
    results["Number of epochs"] = num_epochs
    results["Batch size"] = batch_sizes
    results["Learning rate"] = learning_rates
    results["Layers"] = list_of_layers
    results["Train accuracy"] = train_accuracy_column
    results["Dev accuracy"] = dev_accuracy_column

    return results, costs, best_model


def plot_costs(costs):
    x_values_modified = [x * 100 for x in range(len(costs[0]))]
    for ind, cost in enumerate(costs):
        plt.plot(
            x_values_modified,
            cost,
            marker="o",
            linestyle="-",
            label=f"Model number: {ind}",
        )

    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Costs of models in training")
    plt.legend()
    plt.show()
