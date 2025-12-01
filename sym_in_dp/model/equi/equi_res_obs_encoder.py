import torch
from torchvision import models as vision_models
from escnn import gspaces, nn
from einops import rearrange
from robomimic.models.base_nets import SpatialSoftmax
from sym_in_dp.model.common.module_attr_mixin import ModuleAttrMixin
import sym_in_dp.model.vision.crop_randomizer as dmvc
from sym_in_dp.model.equi.equi_encoder import EquiResBlock
from sym_in_dp.model.common.rotation_transformer import RotationTransformer


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class ResNet18Encoder(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        net = vision_models.resnet18(norm_layer=torch.nn.BatchNorm2d)
        self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size//2)

    def forward(self, ih):
        batch_size = ih.shape[0]
        return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)

class EquivariantResEncoder76Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)
        # 76x76
        self.conv = torch.nn.Sequential(
            # 76x76
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, 8 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 72x72
            nn.ReLU(nn.FieldType(self.group, 8 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, 8, 8, initialize=True),
            EquiResBlock(self.group, 8, 8, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 8 * [self.group.regular_repr]), 2),
            # 36x36
            EquiResBlock(self.group, 8, 16, initialize=True),
            EquiResBlock(self.group, 16, 16, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 16 * [self.group.regular_repr]), 2),
            # 18x18
            EquiResBlock(self.group, 16, 32, initialize=True),
            EquiResBlock(self.group, 32, 32, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 9x9
            EquiResBlock(self.group, 32, 64, initialize=True),
            EquiResBlock(self.group, 64, 64, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 3),
            # 3x3
            nn.R2Conv(
                nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            # 1x1
        )

    def forward(self, x) -> nn.GeometricTensor:
        batch_size = x.shape[0]
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x).tensor.reshape(batch_size, -1)

class EquivariantObsEncIHOnly(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        initialize=True,
    ):
        super().__init__()
        self.enc_ih = EquivariantResEncoder76Cyclic(obs_shape[0], 128, initialize)
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 
        
    def forward(self, nobs):
        ih = nobs["robot0_eye_in_hand_image"]
        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        batch_size = ih.shape[0]
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        ih = self.crop_randomizer(ih)

        ih_out = self.enc_ih(ih)
        out = torch.cat([ih_out, ee_pos, ee_quat, ee_q], dim=1)
        return rearrange(out, "(b t) d -> b t d", b=batch_size)

if __name__ == "__main__":
    enc = EquivariantResEncoder76Cyclic(2, 64, initialize=True)
    x = torch.randn(1, 2, 76, 76)
    out = enc(x)
    print(out.shape)