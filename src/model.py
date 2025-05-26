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

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),     # 224×224 → 224×224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 224×224 → 112×112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 112×112 → 112×112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 112×112 → 56×56       

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 56×56 → 56×56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 56×56 → 28×28     

            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 28×28 → 28×28   
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 28×28 → 14×14     
        )

        # Collapse HxW to 1x1 to avoid hardcoding spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Final linear layer
        self.classifier = nn.Linear(128, num_classes)


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
