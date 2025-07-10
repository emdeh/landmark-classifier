import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.6):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(3,  32),  nn.MaxPool2d(2),   # 224 -> 112
            conv_block(32, 64),  nn.MaxPool2d(2),   # 112-> 56
            conv_block(64,128),  nn.MaxPool2d(2),   # 56 -> 28
            conv_block(128,256), nn.MaxPool2d(2),   # 28 -> 14
            conv_block(256,512), nn.MaxPool2d(2),   # 14 -> 7
        )

        self.avgpool   = nn.AdaptiveAvgPool2d((1,1))
        self.dropout   = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)        # (B,512,7,7)
        x = self.avgpool(x)         # (B,512,1,1)
        x = torch.flatten(x, 1)     # (B,512)
        x = self.dropout(x)
        return self.classifier(x)   # (B,num_classes)



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
