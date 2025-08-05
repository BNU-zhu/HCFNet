import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, 1, H, W]
        # targets: [N, 1, H, W]
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """
    Tversky Loss, a generalization of Dice Loss
    By adjusting alpha and beta, FN or FP can be penalized.
    - alpha=beta=0.5 -> Dice Loss
    - alpha > beta -> Punish FP
    - beta > alpha -> Punish FN 
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [N, 1, H, W]
        # targets: [N, 1, H, W]
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        # True Positives, False Positives & False Negatives
        TP = torch.sum(probs * targets, dim=1)
        FP = torch.sum(probs * (1 - targets), dim=1)
        FN = torch.sum((1 - probs) * targets, dim=1)
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return (1 - tversky_index).mean()


class AreaLossModule(nn.Module):
    """region Task Loss Module (Cross Entropy + Dice)"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = TverskyLoss(alpha=0.5, beta=0.5) # Dice is Tversky with alpha=beta=0.5

    def forward(self, logits, targets_idx):
        # logits: [N, 2, H, W]
        # targets_idx: [N, H, W]
        loss_ce = self.ce_loss(logits, targets_idx)
        
        targets_fg = (targets_idx == 1).unsqueeze(1).float() # [N, 1, H, W]
        logits_fg = logits[:, 1, ...].unsqueeze(1) # [N, 1, H, W]
        loss_dice = self.dice_loss(logits_fg, targets_fg)
        
        return loss_ce + loss_dice

class EnhancedBoundaryLossModule(nn.Module):
    """Boundary task loss module"""
    def __init__(self, focal_alpha=0.5, focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.focal_loss = BinaryFocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, logits, targets):
        # logits: [N, 1, H, W]
        # targets: [N, 1, H, W]
        loss_focal = self.focal_loss(logits, targets)
        loss_tversky = self.tversky_loss(logits, targets)
        return loss_focal + loss_tversky


class LossForHCFNet_Advanced(nn.Module):
    """
    Combined Multi-Component Loss for HCFNet
    - Task balancing: automatic balancing of loss scales using uncertainty-based weight adaptation
    """
    def __init__(self, deep_supervision_weights=[0.3, 0.3, 0.4, 0.5, 0.6, 1.0]):
        super().__init__()
        print("Initializing Advanced Loss for HCFNet with Automatic Balancing...")
        
        if len(deep_supervision_weights) != 6:
            raise ValueError("deep_supervision_weights must have 6 elements.")
            
        self.deep_supervision_weights = deep_supervision_weights
        
        self.criterion_area = AreaLossModule()
        self.criterion_boundary = EnhancedBoundaryLossModule()
        
        # Create learnable log_sigma parameters for automatic task weight balancing
        self.log_sigma_area = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_sigma_boundary = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, outputs_area, outputs_boundary_list, targets_area, targets_boundary):
        """
        Args:
            model_outputs (list): output of the network -> [reg_logits, bou_logits_list]
            targets (dict): truth label -> {'area': LongTensor, 'boundary': FloatTensor}
        """

        targets_boundary = targets_boundary.float()

        raw_loss_area = self.criterion_area(outputs_area, targets_area)


        raw_loss_boundary = 0
        individual_boundary_losses = []
        for i, pred_bou_logits in enumerate(outputs_boundary_list):
            loss_i = self.criterion_boundary(pred_bou_logits, targets_boundary)
            individual_boundary_losses.append(loss_i)
            raw_loss_boundary += self.deep_supervision_weights[i] * loss_i

        # Apply uncertainty weighting to automatically balance losses
        precision_area = torch.exp(-self.log_sigma_area)
        final_loss_area = precision_area * raw_loss_area + self.log_sigma_area
        
        precision_boundary = torch.exp(-self.log_sigma_boundary)
        final_loss_boundary = precision_boundary * raw_loss_boundary + self.log_sigma_boundary
        
        total_loss = final_loss_area + final_loss_boundary
        #print('final_loss_area:', final_loss_area)
        #print('final_loss_boundary:', final_loss_boundary)
        return total_loss
        
        