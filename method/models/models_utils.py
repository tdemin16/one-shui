import timm
from method.models.resnet import resnet18


def get_model(model_name: str, num_classes: int, size: int, pretrained: bool = True):
    if model_name == "resnet18" and size == 32:
        return resnet18(num_classes=num_classes)
    elif "vit" in model_name or "resnet" in model_name:
        return timm.create_model(
            model_name, num_classes=num_classes, pretrained=pretrained
        )
