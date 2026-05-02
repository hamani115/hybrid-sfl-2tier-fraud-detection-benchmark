from collections import OrderedDict

import torch
import torch.nn as nn


def fraud_mlp(
    pretrained,
    num_classes,
    partition,
    last_client_layer,
    input_dim=30,
    dropout=0.2,
):
    """
    Tabular MLP for ULB credit card fraud detection.

    Full model:
        block1: input_dim -> 64
        block2: 64 -> 32
        block3: 32 -> 16
        head:   16 -> num_classes

    Supported partitions:
        partition=None
            full model

        partition="client"
            layers from block1 up to last_client_layer

        partition="server"
            layers after last_client_layer up to head
    """

    _ = pretrained  # kept only to match the repo's architecture function style

    layers = OrderedDict(
        {
            "block1": nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
            "block2": nn.Sequential(
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
            "block3": nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
            ),
            "head": nn.Linear(16, num_classes),
        }
    )

    layer_order = ["block1", "block2", "block3", "head"]

    if partition is None:
        assert last_client_layer is None
        selected_layers = layer_order

    elif partition == "client":
        if last_client_layer not in layer_order:
            raise ValueError(
                f"Invalid last_client_layer={last_client_layer}. "
                f"Choose one of {layer_order}."
            )
        end_idx = layer_order.index(last_client_layer)
        selected_layers = layer_order[: end_idx + 1]

    elif partition == "server":
        if last_client_layer not in layer_order:
            raise ValueError(
                f"Invalid last_client_layer={last_client_layer}. "
                f"Choose one of {layer_order}."
            )
        start_idx = layer_order.index(last_client_layer) + 1
        selected_layers = layer_order[start_idx:]

    else:
        raise ValueError(f"Unsupported partition={partition} for fraud_mlp")

    if len(selected_layers) == 0:
        model = nn.Identity()
    else:
        model = nn.Sequential(
            OrderedDict((name, layers[name]) for name in selected_layers)
        )

    model.is_complete_model = partition is None or (
        partition == "client" and last_client_layer == "head"
    )

    return model


if __name__ == "__main__":
    x = torch.randn(8, 30)

    client = fraud_mlp(
        pretrained=False,
        num_classes=2,
        partition="client",
        last_client_layer="block1",
    )

    server = fraud_mlp(
        pretrained=False,
        num_classes=2,
        partition="server",
        last_client_layer="block1",
    )

    full = fraud_mlp(
        pretrained=False,
        num_classes=2,
        partition=None,
        last_client_layer=None,
    )

    assert server(client(x)).shape == full(x).shape
    print("fraud_mlp ok")