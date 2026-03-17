"""IMPORT PACKAGES"""
import torch.nn as nn
import torch

# Import helper functions from other files
from dinov3.models.vision_transformer import vit_base
from dinov2.models.vision_transformer import vit_base_14
import torch.nn.functional as F

class Model_CLS(nn.Module):
    def __init__(self, opt, inference=False):
        super(Model_CLS, self).__init__()

        # Define the variations that can occur in forward
        self.multi_features = False
        self.features = False

        # Define Backbone architecture
        if opt.backbone == 'DINOv3':
            self.backbone = vit_base(
                    layerscale_init=1.0e-5,
                    mask_k_bias=True,
                    untie_global_and_local_cls_norm=True,
                    n_storage_tokens=4,
                    pos_embed_rope_rescale_coords=2,
                )
            if opt.weights == 'GastroDINO':
                state_dict = torch.load('./weights/Gastro231k.pth', weights_only=False)
            elif opt.weights == 'DINOv3':
                state_dict = torch.load('./weights/dinov3_vitb16_pretrain_lvd1689m.pth', weights_only=False)
            self.backbone.load_state_dict(state_dict, strict=False)

        elif opt.backbone == 'DINOv2':
            self.backbone = vit_base_14(img_size=opt.imagesize)

    def forward(self, img):
        cls = self.backbone(img)
        return cls