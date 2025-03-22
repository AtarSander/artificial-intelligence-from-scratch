from utils import timer
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class TorchModel(nn.Module):
    def __init__(self, layers: list[tuple[tuple[int, int], str]]) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(nn.Linear(layer[0][0], layer[0][1]))
        self.init_weights()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self.cuda()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.flatten(x, start_dim=1)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).float()
        return self(x).cpu().detach().numpy()

    def init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)


class MnistDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_list: np.ndarray,
        img_array: np.ndarray,
        transform=None,
        target_transform=None,
    ):
        self.img_labels = annotations_list
        self.images = img_array
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.images[idx]).float()
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


@timer
def torch_train(
    dataset: MnistDataset,
    model: TorchModel,
    learning_rate: float,
    epochs: int,
    batch_size: int,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        for image, label in dataloader:
            optimizer.zero_grad()
            y_pred = model((image / 255.0).to(device))
            loss = criterion(y_pred, label.to(device))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_sum += loss.item()
        losses.append(loss_sum / len(dataloader))
        if epoch % 10 == 0:
            print(f"Epoch:{epoch}, Loss: {loss_sum / len(dataloader)}")
    return losses
