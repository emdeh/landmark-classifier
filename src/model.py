import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        # Helper block: Conv > BN > ReLU
        def cbr(inp, out):
            return nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            cbr(3,   32), cbr(32,  32), nn.MaxPool2d(2),   # 224 → 112
            cbr(32,  64), cbr(64,  64), nn.MaxPool2d(2),   # 112 → 56
            cbr(64, 128), cbr(128,128), nn.MaxPool2d(2),   # 56  → 28
            cbr(128,256), cbr(256,256), nn.MaxPool2d(2),   # 28  → 14
        )

        self.avgpool   = nn.AdaptiveAvgPool2d(1)           # → 1 × 1
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.features(x)                                # (batch, 128, 14, 14)
        x = self.avgpool(x)                                 # (batch, 128, 1, 1)     
        x = torch.flatten(x, 1)                             # (batch, 128)
        x = self.dropout(x)                                      
        x = self.classifier(x)                              # (batch, num_classes)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
