import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size
        return dice_loss

def dice_coefficient(preds, targets, epsilon=1e-6):
    """
    计算 Dice 系数

    参数:
        preds (Tensor): 预测的分割图像张量，形状为 (N, H, W) 或 (N, C, H, W)
        targets (Tensor): 真实的分割图像张量，形状与 preds 相同
        epsilon (float): 防止除零错误的一个小常量
    返回:
        float: Dice 系数
    """
    # 确保张量是二值化的
    preds = (preds > 0.5)
    # targets = targets > 0.5
    num = preds.size(0)
    # 将预测和目标张量展平
    preds_flat = preds.view(num, -1)
    targets_flat = targets.view(num, -1)

    # 计算交集和并集
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    # 计算 Dice 系数
    dice = 2. * intersection / (union + epsilon)

    return dice

class Metrics():
    
    def __init__(self):

        self.dice_corr = dice_coefficient
        self.dice_loss = DiceLoss()
        self.bceloss = torch.nn.BCELoss(reduction='mean')
        self.sig = nn.Sigmoid()
        
    def loss(self, pred, label, sigmoid=True):
        if sigmoid == True: pred = self.sig(pred)
        loss_value = self.bceloss(pred, label) + self.dice_loss(pred, label)
        return loss_value
    
    def dsc(self, pred, label, sigmoid=True):
        if sigmoid == True: pred = self.sig(pred)
        dsc = self.dice_corr(pred, label)
        return dsc