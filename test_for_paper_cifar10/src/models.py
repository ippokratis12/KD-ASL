import torch
import torch.nn as nn
import torchvision
from utils import set_global_seed

class ResNet18Backbone(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        try:
            base = torchvision.models.resnet18(weights=None)
        except TypeError:
            base = torchvision.models.resnet18(pretrained=False)

        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.features = nn.Sequential(*list(base.children())[:-1])
        in_feats = base.fc.in_features
        self.fc = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

    def get_features(self, x):
        x = self.features(x).flatten(1)
        return x

def resnet18_backbone(n_classes=10):
    return ResNet18Backbone(n_classes=n_classes)

class TeacherEnsemble(nn.Module):
    def __init__(self, n=5, n_classes=10, base_seed=42):
        super().__init__()
        self.models = nn.ModuleList()
        for i in range(n):
            local_seed = base_seed + i
            set_global_seed(local_seed)
            self.models.append(resnet18_backbone(n_classes=n_classes))

    def forward(self, x):
        return torch.stack([m(x) for m in self.models], 0).mean(0)

    def get_features(self, x):
        return torch.stack([m.get_features(x) for m in self.models], 0).mean(0)

class Encoder32(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        try:
            base = torchvision.models.resnet18(weights=None)
        except TypeError:
            base = torchvision.models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

class Decoder32(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64,  3,4,2,1), nn.Sigmoid()
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0),512,1,1))

class CAE2(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = Encoder32(latent_dim)
        self.decoder = Decoder32(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class General_Class_Network(nn.Module):
    def __init__(self, input_size=10, output_size=10, noise_std=1e-4):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.data = torch.eye(input_size, output_size) + noise_std * torch.randn(input_size, output_size)

    def forward(self, x):
        return self.linear(x)