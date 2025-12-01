from typing import Dict
import copy
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from sym_in_dp.model.common.normalizer import LinearNormalizer
from sym_in_dp.policy.base_image_policy import BaseImagePolicy
from sym_in_dp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from sym_in_dp.model.diffusion.mask_generator import LowdimMaskGenerator
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
from sym_in_dp.model.vision.rot_randomizer import RotRandomizer
from sym_in_dp.model.common.rotation_transformer import RotationTransformer
from sym_in_dp.model.equi.equi_res_obs_encoder import EquivariantObsEncIHOnly


class DiffusionEquiEncRelTrajPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']

        del obs_shape_meta['agentview_image']

        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # n_hidden = 16
        # self.obs_encoder = EquivariantObsEnc(
        #     obs_shape=obs_shape_meta['robot0_eye_in_hand_image']['shape'], 
        #     crop_shape=crop_shape, 
        #     n_hidden=n_hidden, 
        #     N=8)

        # # create diffusion model
        # obs_feature_dim = 3 * 3 * n_hidden * 8 + 3 + 6 + 2

        self.obs_encoder = EquivariantObsEncIHOnly(
            obs_shape=obs_shape_meta['robot0_eye_in_hand_image']['shape'], 
            crop_shape=crop_shape
            )

        # create diffusion model
        obs_feature_dim = 128 * 8 + 3 + 4 + 2

        # obs_feature_dim = 128 * 8
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            # global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.sixd_to_mat = RotationTransformer(
            from_rep='rotation_6d',
            to_rep='matrix',
        )
        self.quat_to_mat = RotationTransformer(
            from_rep='quaternion',
            to_rep='matrix',
        )

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
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

            nobs_features = self.obs_encoder(nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:

            nobs_features = self.obs_encoder(nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        xyz = action_pred[:, :, :3]
        rot_6d = action_pred[:, :, 3:9]
        rot = self.sixd_to_mat.forward(rot_6d)
        rel_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        rel_T[:, :, :3, :3] = rot
        rel_T[:, :, :3, 3] = xyz

        cur_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        cur_rot = self.quat_to_mat.forward(obs_dict['robot0_eef_quat'][:, :, [3, 0, 1, 2]][:, -1:])
        cur_trans = obs_dict['robot0_eef_pos'][:, -1:]
        cur_T[:, :, :3, :3] = cur_rot
        cur_T[:, :, :3, 3] = cur_trans

        abs_T = torch.matmul(cur_T, rel_T)
        abs_rot = abs_T[:, :, :3, :3]
        abs_xyz = abs_T[:, :, :3, 3]
        abs_rot_6d = self.sixd_to_mat.inverse(abs_rot)
        abs_action = torch.cat([abs_xyz, abs_rot_6d, action_pred[:, :, 9:]], dim=-1)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = abs_action[:,start:end]
        
        result = {
            'action': action,
            'action_pred': abs_action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        batch = copy.deepcopy(batch)
        assert 'valid_mask' not in batch
        del batch['obs']['agentview_image']
        # First convert to relative coordinates
        abs_xyz = batch['action'][:, :, :3]
        abs_6d = batch['action'][:, :, 3:9]
        abs_T = torch.eye(4).repeat(batch['action'].shape[0], batch['action'].shape[1], 1, 1).to(self.device)
        abs_T[:, :, :3, :3] = self.sixd_to_mat.forward(abs_6d)
        abs_T[:, :, :3, 3] = abs_xyz
        cur_T = torch.eye(4).repeat(batch['action'].shape[0], batch['action'].shape[1], 1, 1).to(self.device)
        cur_T[:, :, :3, :3] = self.quat_to_mat.forward(batch['obs']['robot0_eef_quat'][:, :, [3, 0, 1, 2]][:, -1:])
        cur_T[:, :, :3, 3] = batch['obs']['robot0_eef_pos'][:, -1:]
        rel_T = cur_T.inverse() @ abs_T
        rel_xyz = rel_T[:, :, :3, 3]
        rel_6d = self.sixd_to_mat.inverse(rel_T[:, :, :3, :3])
        rel_action = torch.cat([rel_xyz, rel_6d, batch['action'][:, :, 9:]], dim=-1)

        # # Debug prints for original case
        # print("\nOriginal case:")
        # print("cur_T:", cur_T[0, 0])
        # print("abs_T:", abs_T[0, 0])
        # print("rel_T:", rel_T[0, 0])

        # # Then normalize and apply rotation
        # # nobs = self.normalizer.normalize(batch['obs'])
        # # nactions = self.normalizer['action'].normalize(batch['action'])
        # rotated_obs, rotated_action = self.rot_randomizer(batch['obs'], batch['action'])
        # # rotated_obs = self.normalizer.unnormalize(roted_nobs)
        # # rotated_action = self.normalizer['action'].unnormalize(roted_naction)

        # # Convert back to absolute coordinates
        # abs_xyz = rotated_action[:, :, :3]
        # abs_6d = rotated_action[:, :, 3:9]
        # abs_T = torch.eye(4).repeat(rotated_action.shape[0], rotated_action.shape[1], 1, 1).to(self.device)
        # abs_T[:, :, :3, :3] = self.sixd_to_mat.forward(abs_6d)
        # abs_T[:, :, :3, 3] = abs_xyz
        # cur_T = torch.eye(4).repeat(rotated_action.shape[0], rotated_action.shape[1], 1, 1).to(self.device)
        # cur_T[:, :, :3, :3] = self.quat_to_mat.forward(rotated_obs['robot0_eef_quat'][:, :, [3, 0, 1, 2]][:, -1:])
        # cur_T[:, :, :3, 3] = rotated_obs['robot0_eef_pos'][:, -1:]

        # # Let RotRandomizer2 handle the rotation
        # rel_T = cur_T.inverse() @ abs_T
        # rel_xyz = rel_T[:, :, :3, 3]
        # rel_6d = self.sixd_to_mat.inverse(rel_T[:, :, :3, :3])
        # rel_action_1 = torch.cat([rel_xyz, rel_6d, rotated_action[:, :, 9:]], dim=-1)

        # # Debug prints for rotated case
        # print("\nRotated case:")
        # print("cur_T:", cur_T[0, 0])
        # print("abs_T:", abs_T[0, 0])
        # print("rel_T:", rel_T[0, 0])

        # # Debug prints for poses
        # print("\nPoses:")
        # print("Original gripper pose:", batch['obs']['robot0_eef_pos'][0, -1])
        # print("Original target pose:", batch['action'][0, 0, :3])
        # print("Original relative:", rel_action[0, 0, :3])
        # print("Rotated gripper pose:", rotated_obs['robot0_eef_pos'][0, -1])
        # print("Rotated target pose:", rotated_action[0, 0, :3])
        # print("Rotated relative:", rel_action_1[0, 0, :3])

        batch['action'] = rel_action
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:

            nobs_features = self.obs_encoder(nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:

            nobs_features = self.obs_encoder(nobs)
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
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

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
