import torchvision
from torch import nn


RESCALE_SIZE = 224, 224
N_CLASSES = 400


transform = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(size=RESCALE_SIZE),
                        torchvision.transforms.ToTensor(),
            ])

model = torchvision.models.vgg19_bn()
num_features = 25088
model.classifier = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=N_CLASSES, bias=True),
        )