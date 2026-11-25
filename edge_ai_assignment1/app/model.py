"""model.py -- the network, shared by train.py and the server so both build the same thing."""

from torch import nn

from app.sensors import FEATURES


class BatteryNet(nn.Module):
    def __init__(self, hidden=(32, 32)):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def describe(net):
    layers = []
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            macs = layer.in_features * layer.out_features
            params = sum(p.numel() for p in layer.parameters())
            layers.append(
                {
                    "name": f"linear{len(layers) + 1}",
                    "in": layer.in_features,
                    "out": layer.out_features,
                    "params": params,
                    "bytes_f32": 4 * params,
                    "macs": macs,
                }
            )
    total = sum(layer["params"] for layer in layers)
    return {
        "layers": layers,
        "total_params": total,
        "total_bytes_f32": 4 * total,
        "total_bytes_int8": total,
        "total_macs": sum(layer["macs"] for layer in layers),
        "fits_300kb_sram_f32": 4 * total <= 300 * 1024,
        "fits_300kb_sram_int8": total <= 300 * 1024,
    }
