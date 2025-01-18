import torchvision.models as models
import torch.nn as nn
import torch

# 1. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM的输出
        lstm_out, (hidden, cell) = self.lstm(x)
        # 全连接层
        out = self.fc(hidden)
        return out

def get_model(args):
    model = LSTMModel(args.input_size, args.hidden_size, args.output_size)
    model.to(args.device)
    return model 

if __name__ == "__main__":
    model = ViT(image_size=224, patch_size=16, num_classes=100, dim=768, depth=12, heads=12, mlp_dim=3072,
                dropout=0.1)
    print(model)