import torch.nn as nn
from torchvision import models

#Model EfficientNetV2
def build_efficientnetv2(num_classes, pretrained=True):
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model