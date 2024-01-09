from model import Model


def experiment(list_of_layers, learning_rates, data, batch_sizes, num_epochs):
    results = {}
    for i in range(list_of_layers):
        results["Index"] = i
        results["Number of epochs"] = num_epochs[i]
        results["Batch size"] = batch_sizes[i]
        results["Learning rate"] = learning_rates[i]
        results["Layers"] = list_of_layers[i]

        new_model = Model(list_of_layers[i])
        new_model.train(,,learning_rates[i], num_epochs[i], batch_sizes[i], 1, False)
