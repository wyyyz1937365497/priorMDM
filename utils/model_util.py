import torch
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from model.cfg_sampler import wrap_model
from model.comMDM import ComMDM
from model.mdm import MDM
from model.DoubleTake_MDM import doubleTake_MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.model_blending import ModelBlender

def load_model_blending_and_diffusion(args_list, data, device, ModelClass=MDM, DiffusionClass=gd.GaussianDiffusion):
    models = [load_model(args, data, device, ModelClass)[0] for args in args_list]
    model = ModelBlender(models, [1/len(models)]*len(models)) if len(models) > 1 else models[0]
    
    _, diffusion = create_model_and_diffusion(args_list[0], data, DiffusionClass=DiffusionClass)
    return model, diffusion

def load_model(model_path,device, ModelClass=MDM):
    model, diffusion = create_model_and_diffusion(ModelClass=ModelClass)
    model_path = model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model)
    return model, diffusion

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if 't_pos_encoder.pe' in missing_keys:
        missing_keys.remove('t_pos_encoder.pe')
    if 't_pos_encoder.pe' in unexpected_keys:
        unexpected_keys.remove('t_pos_encoder.pe')
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def load_pretrained_mdm(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('multi_person.') for k in missing_keys])


def load_split_mdm(model, state_dict, cutting_point):
    new_state_dict = {}
    orig_trans_prefix = 'seqTransEncoder.'
    for k, v in state_dict.items():
        if k.startswith(orig_trans_prefix):
            orig_layer = int(k.split('.')[2])
            orig_suffix = '.'.join(k.split('.')[3:])
            target_split = 'seqTransEncoder_start.' if orig_layer < cutting_point else 'seqTransEncoder_end.'
            target_layer = orig_layer if orig_layer < cutting_point else orig_layer - cutting_point
            new_k = target_split + 'layers.' + str(target_layer) + '.' + orig_suffix
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('multi_person.') for k in missing_keys])


def create_model_and_diffusion(ModelClass=MDM, DiffusionClass=SpacedDiffusion):
    model = ModelClass(**get_model_args())
    diffusion = create_gaussian_diffusion(DiffusionClass)
    return model, diffusion

def get_model_args():

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = 'text' 
    num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    data_rep = 'hml_vec'
    njoints = 263
    nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': 0.1, 'action_emb': action_emb, 'arch': 'trans_enc',
            'emb_trans_dec': False, 'clip_version': clip_version, 'dataset': 'humanml',
            'diffusion-steps': 50, 'batch_size': 64, 'use_tta': False,
            'trans_emb': False, 'concat_trans_emb': False }


def create_gaussian_diffusion(DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 50
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    
    print(f"number of diffusion-steps: {steps}")

    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    multi_train_mode = None

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma #  如果learn_sigma为False，则返回gd.ModelVarType.FIXED_RANGE
            else gd.ModelVarType.LEARNED_RANGE #  否则返回gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        batch_size=64,
        multi_train_mode=multi_train_mode,
    )