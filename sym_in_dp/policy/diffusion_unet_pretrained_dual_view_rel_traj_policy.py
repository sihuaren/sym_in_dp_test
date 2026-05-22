from typing import Dict
import copy

import torch
import torch.nn.functional as F
from einops import reduce

from sym_in_dp.common.pytorch_util import dict_apply
from sym_in_dp.model.vision.timm_obs_encoder import (
    C8EquivariantTimmObsEncoder,
    TimmObsEncoder,
)
from sym_in_dp.policy.diffusion_unet_pretrained_rel_traj_policy import (
    DiffusionUnetPretrainedRelTrajPolicy,
)


class DiffusionUnetPretrainedDualViewRelTrajPolicy(
        DiffusionUnetPretrainedRelTrajPolicy):
    """Pretrained relative-trajectory policy that keeps both configured views.

    The obs encoder concatenates RGB features in sorted key order, so
    ``agentview_image`` is placed before ``robot0_eye_in_hand_image`` when both
    keys are present in ``shape_meta``.
    """

    def _obs_encoder_handles_temporal_obs(self):
        return isinstance(
            self.obs_encoder,
            (TimmObsEncoder, C8EquivariantTimmObsEncoder),
        )

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            if self._obs_encoder_handles_temporal_obs():
                this_nobs = nobs
            else:
                this_nobs = dict_apply(
                    nobs,
                    lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
                )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        if self.input_action_space == 'relative_axis_angle':
            action = action_pred[:, start:end]
            result = {
                'action': action,
                'action_pred': action_pred
            }
            return result

        xyz = action_pred[:, :, :3]
        rot_6d = action_pred[:, :, 3:9]
        rot = self.sixd_to_mat.forward(rot_6d)
        rel_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        rel_T[:, :, :3, :3] = rot
        rel_T[:, :, :3, 3] = xyz

        cur_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        cur_rot = self.quat_to_mat.forward(
            obs_dict['robot0_eef_quat'][:, :, [3, 0, 1, 2]][:, -1:]
        )
        cur_trans = obs_dict['robot0_eef_pos'][:, -1:]
        cur_T[:, :, :3, :3] = cur_rot
        cur_T[:, :, :3, 3] = cur_trans

        abs_T = torch.matmul(cur_T, rel_T)
        abs_rot = abs_T[:, :, :3, :3]
        abs_xyz = abs_T[:, :, :3, 3]
        abs_rot_6d = self.sixd_to_mat.inverse(abs_rot)
        abs_action = torch.cat([abs_xyz, abs_rot_6d, action_pred[:, :, 9:]], dim=-1)

        # get action
        action = abs_action[:, start:end]

        result = {
            'action': action,
            'action_pred': abs_action
        }
        return result

    # ========= training ============
    def compute_loss(self, batch):
        # normalize input
        batch = copy.deepcopy(batch)
        assert 'valid_mask' not in batch

        if self.input_action_space == 'absolute_6d':
            # Convert absolute target poses to relative actions before diffusion.
            abs_xyz = batch['action'][:, :, :3]
            abs_6d = batch['action'][:, :, 3:9]
            abs_T = torch.eye(
                4,
                device=batch['action'].device,
                dtype=batch['action'].dtype
            ).repeat(batch['action'].shape[0], batch['action'].shape[1], 1, 1)
            abs_T[:, :, :3, :3] = self.sixd_to_mat.forward(abs_6d)
            abs_T[:, :, :3, 3] = abs_xyz
            cur_T = torch.eye(
                4,
                device=batch['action'].device,
                dtype=batch['action'].dtype
            ).repeat(batch['action'].shape[0], batch['action'].shape[1], 1, 1)
            cur_T[:, :, :3, :3] = self.quat_to_mat.forward(
                batch['obs']['robot0_eef_quat'][:, :, [3, 0, 1, 2]][:, -1:]
            )
            cur_T[:, :, :3, 3] = batch['obs']['robot0_eef_pos'][:, -1:]
            rel_T = cur_T.inverse() @ abs_T
            rel_xyz = rel_T[:, :, :3, 3]
            rel_6d = self.sixd_to_mat.inverse(rel_T[:, :, :3, :3])
            batch['action'] = torch.cat(
                [rel_xyz, rel_6d, batch['action'][:, :, 9:]],
                dim=-1
            )

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            if self._obs_encoder_handles_temporal_obs():
                this_nobs = nobs
            else:
                this_nobs = dict_apply(
                    nobs,
                    lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
                )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
