"""CNN totalmente convolucional para refinar SRP-PHAT."""

from __future__ import annotations


def build_model(dilations: tuple[int, ...] | None = None):
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch é necessário para construir o modelo") from exc

    dilation_values = tuple(dilations or (1, 1, 1, 1))
    if len(dilation_values) != 4 or any(value < 1 for value in dilation_values):
        raise ValueError("dilations deve conter quatro inteiros positivos")

    class ConvLocalizationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=dilation_values[0], dilation=dilation_values[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=dilation_values[1], dilation=dilation_values[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=dilation_values[2], dilation=dilation_values[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=dilation_values[3], dilation=dilation_values[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )

        def forward(self, inputs):
            return self.network(inputs)

    return ConvLocalizationNet()

