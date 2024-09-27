import torchvision.models as models
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

def get_model(args):
    model = smp.Unet(
            encoder_name="resnet18",        # Choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            decoder_channels=[16, 32, 64, 128, 256],
            encoder_weights="imagenet",     # Use pre-trained weights from ImageNet
            in_channels=3,                  # Input channels (RGB images have 3 channels)
            classes=1,                      # Number of output classes (1 for binary segmentation)
        ).to(args.device)
    
    return model

if __name__ == "__main__":
    model = ViT(image_size=224, patch_size=16, num_classes=100, dim=768, depth=12, heads=12, mlp_dim=3072,
                dropout=0.1)
    print(model)