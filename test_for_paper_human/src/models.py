# -*- coding: utf-8 -*-
# src/models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision
from utils import seed_and_optionally_deterministic

# ResNet-18 backbone (64x64 -> keep default stem) 
                         
class ResNet18Backbone(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        try:
            base = torchvision.models.resnet18(weights=None)
        except TypeError:
            base = torchvision.models.resnet18(pretrained=False)

        self.features = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        in_feats = base.fc.in_features
        self.fc = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

    def get_features(self, x):
        x = self.features(x).flatten(1)
        return x

def resnet18_backbone(n_classes=2):
    return ResNet18Backbone(n_classes=n_classes)


# Deep Ensemble with per-teacher seeds (DETERMINISTIC)
                     
class TeacherEnsemble(nn.Module):
    def __init__(self, n=5, n_classes=2, base_seed=42):
        super().__init__()
        self.models = nn.ModuleList()
        for i in range(n):
            local_seed = base_seed + i
            # teacher init should be deterministic
            seed_and_optionally_deterministic(local_seed, deterministic=True)
            self.models.append(resnet18_backbone(n_classes=n_classes))

    def forward(self, x):
        return torch.stack([m(x) for m in self.models], 0).mean(0)

    def get_features(self, x):
        return torch.stack([m.get_features(x) for m in self.models], 0).mean(0)



# CAE-2 & GCN                                                             
class Encoder32(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        try: base = torchvision.models.resnet18(weights=None)
        except TypeError: base = torchvision.models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

class Decoder32(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d( 64, 64,4,2,1), nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0),512,1,1))

class CAE2(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder32(latent_dim)
        self.decoder = Decoder32(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class General_Class_Network(nn.Module):
    def __init__(self, input_size=2, output_size=2, noise_std=1e-4):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.data = torch.eye(input_size, output_size) + noise_std * torch.randn(input_size, output_size)
    def forward(self, x):
        return self.linear(x)