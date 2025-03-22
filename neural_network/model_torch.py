import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class TorchModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=784, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=32)
        if device.type == "cuda":
            self.cuda()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


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

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


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
            print(f"Loss:{loss}")
            loss.backward()
            optimizer.step()
            loss_sum += loss
        losses.append(loss_sum / len(dataloader))
        if epoch % 10 == 0:
            print(f"Epoch:{epoch}, Loss: {loss_sum / len(dataloader)}")
    return losses
