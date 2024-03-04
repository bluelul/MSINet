import torch.nn as nn
import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss, cosine_dist
from .spatial_align_loss import SpatialAlignLoss
from .recallatk_loss import RecallatK

class ReIDLoss(nn.Module):
    """Build the loss function for ReID tasks. 
    """
    def __init__(self, args, num_classes):
        super(ReIDLoss, self).__init__()

        self.triplet = TripletLoss(args.margin, normalize_feature=True)
        self.softmax = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.recallatk = RecallatK(batch_size=args.batch_size, samples_per_class=args.num_instance)
        if args.sam_mode != 'none':
            self.spatial = SpatialAlignLoss(args.sam_mode)
            self.sam_mode = args.sam_mode
            self.sam_ratio = args.sam_ratio
        else:
            self.sam_mode = None

    def forward(self, feats, logits, sam_logits, target, sam=False):
        recallatk_loss = self.recallatk(logits, target)
        # return recallatk_loss
        triplet_loss = self.triplet(feats, target)[0]
        softmax_loss = self.softmax(logits, target)
        if self.sam_mode != None and sam:
            spatial_loss = self.spatial(sam_logits, target)
            return softmax_loss + triplet_loss + spatial_loss * self.sam_ratio
        else:
            return softmax_loss + triplet_loss


def build_criterion(args, num_classes):
    return ReIDLoss(args, num_classes)
