# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import pickle
from model.DoubleTake_MDM import doubleTake_MDM
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len
import random
from fastapi import FastAPI, Body

app = FastAPI()

model=None
diffusion=None
dist_util.setup_dist(0)
def prepare_diffusion():
        # 打印生成样本
    print(f"generating samples")
    global model,diffusion,dist_util
    seed=random.randint(0, 100)
    fixseed(seed)
    print(f"seed: {seed}")
    model_path = ".//save//humanml_enc_512_50steps//model000750000.pt"
    dist_util.setup_dist(0)
    print(f"device: {dist_util.dev()}")

    # 创建模型和扩散
    print("Creating model and diffusion...")
    model, diffusion = load_model(model_path, dist_util.dev(), ModelClass=doubleTake_MDM)

@app.post("/generate")
def main(texts:list=Body(...)):
    # 打印生成样本
    global model,diffusion,dist_util
    print(f"generating samples")
    # 输出路径
    out_path = './output'
    # 帧率
    fps = 30 
    # 帧数
    n_frames = 150
    # 设置分布式
    dataset_name = 'humanml'
    num_repetitions=1
    handshake_size=20
    blend_len=10
    sample_gt=False
    prepare_diffusion()
    # 设置样本数量
    num_samples = len(texts)
    print(f"num_samples: {num_samples}")
    print(f"input_text: {texts}")
    #["A person walks", "A person sits while crossing legs"]
    # 设置批量大小
    batch_size = num_samples  # Sampling a single batch from the testset, with exactly num_samples

    total_num_samples = num_samples * num_repetitions

    model_kwargs = {'y': {
            'mask': torch.ones((len(texts), 1, 1, 196)), # 196 is humanml max frames number
            'lengths': torch.tensor([n_frames]*len(texts)),
            'text': texts,
            'tokens': [''],
            'scale': torch.ones(len(texts))*2.5
        }}

    # 初始化变量
    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    # 重复采样
    for rep_i in range(num_repetitions):
        # 打印采样
        print(f'### Sampling [repetitions #{rep_i}]')
        # 如果引导参数不为1
        # 设置引导参数
        model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * 2.5
        # 设置模型参数
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        # 设置最大和最小长度
        max_arb_len = model_kwargs['y']['lengths'].max()
        min_arb_len = 2 * handshake_size + 2*blend_len + 10

        # 设置长度
        for ii, len_s in enumerate(model_kwargs['y']['lengths']):
            if len_s > max_arb_len:
                model_kwargs['y']['lengths'][ii] = max_arb_len
            if len_s < min_arb_len:
                model_kwargs['y']['lengths'][ii] = min_arb_len
        # 采样
        samples_per_rep_list, samples_type = double_take_arb_len(blend_len,handshake_size, diffusion, model, model_kwargs, max_arb_len)

        # 设置步长
        step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
        for ii, len_i in enumerate(model_kwargs['y']['lengths']):
            if ii == 0:
                step_sizes[ii] = len_i
                continue
            step_sizes[ii] = step_sizes[ii-1] + len_i - handshake_size

        # 设置最终帧数
        final_n_frames = step_sizes[-1]

        # 采样
        for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

            # 展开采样
            sample = unfold_sample_arb_len(sample_i, handshake_size, step_sizes, final_n_frames, model_kwargs)

            # Recover XYZ *positions* from HumanML3D vector representation
            n_joints = 22 if sample.shape[1] == 263 else 21
            with open('inv_transform.data', 'rb') as f:
                inv_transform= pickle.load(f) 
            sample = inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'

            all_text += model_kwargs['y'][text_key]
            all_captions += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * batch_size} samples")

    # param update for unfolding visualization
    # out of for rep_i
    old_num_samples = num_samples
    num_samples = 1
    batch_size = 1
    n_frames = final_n_frames

    num_repetitions = num_repetitions

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = [n_frames] * num_repetitions

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    frame_colors = calc_frame_colors(handshake_size, blend_len, step_sizes, model_kwargs['y']['lengths'])
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': num_samples, 'num_repetitions': num_repetitions, 'frame_colors': frame_colors})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    # print(f"saving visualizations to [{out_path}]...")
    # skeleton =  paramUtil.t2m_kinematic_chain
    # for sample_i in range(num_samples):
    #     rep_files = []
    #     for rep_i, samples_type_i in zip(range(num_repetitions), samples_type):
    #         caption = [f'{samples_type_i} {all_text[0]}'] * (model_kwargs['y']['lengths'][0] - int(handshake_size/2))
    #         for ii in range(1, old_num_samples):
    #             caption += [f'{samples_type_i} {all_text[ii]}'] * (int(model_kwargs['y']['lengths'][ii])-handshake_size)
    #         caption += [f'{samples_type_i} {all_text[ii]}'] * (int(handshake_size/2))
    #         length = all_lengths[rep_i*batch_size + sample_i]
    #         motion = all_motions[rep_i*batch_size + sample_i].transpose(2, 0, 1)[:length]
    #         save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
    #         animation_save_path = os.path.join(out_path, save_file)
    #         print(f'[({sample_i}) "{set(caption)}" | Rep #{rep_i} | -> {save_file}]')
    #         plot_3d_motion(animation_save_path, skeleton, motion, dataset=dataset_name, title=caption, fps=fps,
    #                        vis_mode='gt' if sample_gt else 'unfold_arb_len', handshake_size=handshake_size,
    #                        blend_size=blend_len,step_sizes=step_sizes, lengths=model_kwargs['y']['lengths'])
    #         # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
    #         rep_files.append(animation_save_path)

    # abs_path = os.path.abspath(out_path)
    # print(f'[Done] Results are at [{abs_path}]')


def calc_frame_colors(handshake_size, blend_size, step_sizes, lengths):
    for ii, step_size in enumerate(step_sizes):
        if ii == 0:
            frame_colors = ['orange'] * (step_size - handshake_size - blend_size) + \
                           ['blue'] * blend_size + \
                           ['purple'] * (handshake_size // 2)
            continue
        if ii == len(step_sizes) - 1:
            frame_colors += ['purple'] * (handshake_size // 2) + \
                            ['blue'] * blend_size + \
                            ['orange'] * (lengths[ii] - handshake_size - blend_size)
            continue
        frame_colors += ['purple'] * (handshake_size // 2) + ['blue'] * blend_size + \
                        ['orange'] * (lengths[ii] - 2 * handshake_size - 2 * blend_size) + \
                        ['blue'] * blend_size + \
                        ['purple'] * (handshake_size // 2)
    return frame_colors

if __name__ == "__main__":
    prepare_diffusion()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=19256
                )