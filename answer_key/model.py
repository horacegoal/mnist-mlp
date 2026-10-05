import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, in_dim=28 * 28, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


class MLP(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden=(128, 64), num_classes=10, activation="relu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        if activation not in acts:
            raise ValueError(f"unknown activation: {activation}")
        act_cls = acts[activation]

        self.flatten = nn.Flatten()
        dims = [in_dim, *hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), act_cls()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.flatten(x))


def build_model(name, hidden=(128, 64), activation="relu"):
    if name == "linear":
        return LinearModel()
    if name == "mlp":
        return MLP(hidden=hidden, activation=activation)
    raise ValueError(f"unknown model: {name}")
