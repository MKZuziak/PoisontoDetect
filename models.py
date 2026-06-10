import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(dataset: str):
    model = None
    dataset = dataset.strip().lower()
    if dataset == "mnist":
        model = SimpleNetworkMNIST()
    elif dataset == "cifar10":
        model = ResNet18CIFAR10()
    elif dataset == "cifar100":
        model = ResNet18CIFAR100()
    elif dataset == "pathmnist":
        model = ResNet18PathMNIST(n_classes=9)  # PathMNIST has 9 classes
    elif dataset == "eurosat":
        model = ResNet18_EuroSAT()
    elif dataset == "fashionmnist":
        model = FashionCNN()
    else:
        raise NameError("Wrong model name. Please choose from: MNIST, FASHIONMNIST, CIFAR10, CIFAR100, PATHMNIST, EUROSAT.")
    return model

#### FashionMNIST ####


class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)  # GroupNorm for small batch sizes
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#### MNIST ####
class SimpleNetworkMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


#### FMNIST ####
class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(pretrained=False)

        # Modify the first conv layer: 3x3 kernel, stride=1, padding=1
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        # Remove maxpool to keep spatial size (32x32)
        self.model.maxpool = nn.Identity()

        # Replace the final fully connected layer to match CIFAR-10 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


#### CIFAR100 ####
class ResNet18CIFAR100(nn.Module):
    def __init__(self, num_classes=100, dropout_p=0.2):
        super().__init__()
        
        # Load ResNet18 without the final classification head
        self.base_model = models.resnet18(weights=None)  # or weights="IMAGENET1K_V1" if you want to load and adapt pretrained
        
        # Modify the first conv layer to work with 32x32 images
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the first maxpool to preserve resolution
        self.base_model.maxpool = nn.Identity()
        
        # Optionally add dropout before final layer
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


#### PATHMNIST ####
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18PathMNIST(n_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes)


def ResNet34PathMNIST(n_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)


def ResNet50PathMNIST(n_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes)


def ResNet101PathMNIST(n_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=n_classes)


def ResNet152PathMNIST(n_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_blocks=n_classes)


#### EUROSAT ####
def norm_layer(num_channels, num_groups=16):
    """Use GroupNorm instead of BatchNorm (better for FL)."""
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class BasicBlock_eu(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=16):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.gn1 = norm_layer(planes, num_groups=num_groups)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.gn2 = norm_layer(planes, num_groups=num_groups)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(self.expansion * planes, num_groups=num_groups)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_EuroSAT(nn.Module):
    def __init__(self, num_classes=10, num_groups=16):
        super().__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn1 = norm_layer(64, num_groups=num_groups)

        # ResNet-18 layers: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock_eu, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock_eu, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock_eu, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock_eu, 512, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # global average pooling
        self.linear = nn.Linear(512 * BasicBlock_eu.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, num_groups=self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)                # → (B, C, 1, 1)
        out = torch.flatten(out, 1)        # → (B, C)
        out = self.linear(out)             # → (B, num_classes)
        return out