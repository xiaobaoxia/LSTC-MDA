import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm


def smooth_distribution(p, beta=0.5):
    p = np.array(p)
    p_smooth = np.power(p, beta)
    return p_smooth / np.sum(p_smooth)

def mix_with_uniform(p, epsilon=0.1):
    p = np.array(p)
    uniform = np.ones_like(p) / len(p)
    p_smooth = (1 - epsilon) * p + epsilon * uniform
    return p_smooth / np.sum(p_smooth)

def temperature_scaling(p, temperature=1.0):
    p = np.array(p)
    logits = np.log(p + 1e-12)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        # required=True,
                        default='ntu60/xsub',
                        # default='ntu60/xview',
                        # default='ntu120/xsub',
                        choices={'ntu60/xsub', 'ntu60/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=0.8,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        default='work_dir/ntu/cs/SkateFormer_j_tconv_jmda_add_mixup/',
                        # default='work_dir/ntu/cv/SkateFormer_j',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        default='work_dir/ntu/cs/SkateFormer_b_tconv_jmda_add_mixup',
                        # default='work_dir/ntu/cv/SkateFormer_b',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir',
                        # default=None,
                        # default='work_dir/ntu/cs/SkateFormer_jm',
                        default='work_dir/ntu/cs/SkateFormer_jm_tconv_jmda_add_mixup/',
                        )
    parser.add_argument('--bone-motion-dir',
                        # default=None,
                        # default='work_dir/ntu/cs/SkateFormer_bm',
                        default='work_dir/ntu/cs/SkateFormer_bm_tconv_jmda_add_mixup/',
                        )

    arg = parser.parse_args()

    dataset = arg.dataset
    
    if 'ntu60' in arg.dataset:
        arg.alpha = [1.1, 1.0, 0.4, 0.3] # best E4 94.0559% new
        arg.alpha = [1.4, 1.0, 0, 0] # best E2 93.8618% new
        arg.alpha = [1, 0, 0, 0]
    elif 'UCLA' in arg.dataset:
         # ucla
        arg.alpha = [0.4, 0.4, 0.3, 0.2]
    
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu60' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r11 = smooth_distribution(r11, beta=0.05)
            r22 = smooth_distribution(r22, beta=0.05)
            r33 = smooth_distribution(r33, beta=0.05)
            r44 = smooth_distribution(r44, beta=0.05)
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            # smooth the distribution
            r11 = smooth_distribution(r11, beta=0.05)
            r22 = smooth_distribution(r22, beta=0.05)
            r33 = smooth_distribution(r33, beta=0.05)
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
