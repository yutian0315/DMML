import torchvision.models as models
import torch.nn as nn
import torch

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        # 加载预训练的ResNet-18模型，并去掉最后一层全连接层
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # resnet18 = models.resnet18(weights=None)
        self._features = nn.Sequential(*list(resnet18.children())[:-1])
        # 添加一个新的全连接层，将输出特征映射到类别数
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(args):
    model = ResNet18(args.num_class)
    model.to(args.device)
    return model 

if __name__ == "__main__":
    model = ViT(image_size=224, patch_size=16, num_classes=100, dim=768, depth=12, heads=12, mlp_dim=3072,
                dropout=0.1)
    print(model)