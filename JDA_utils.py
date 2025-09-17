import torch
import torch.nn.functional as F

import numpy as np
import random

ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)



def Mix_up(x, y):
    N = x.shape[0]
    if N == 0:
        return x, y
    lam = np.random.beta(1.0, 1.0)
    index = torch.randperm(N).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


def resize_window(data, window):
    N, C, T, V, M = data.shape
    data = data.permute(0, 1, 3, 4, 2).contiguous().view(N * C * V * M, T)
    data = data[None, None, :, :]
    data = torch.nn.functional.interpolate(data, size=(N *  C * V * M, window), mode='bilinear',align_corners=False).squeeze()
    data = data.contiguous().view(N, C, V, M, window).permute(0, 1, 4, 2, 3).contiguous()
    return data

def Mix_T(x, y):
    N, C, T, V, M = x.shape
    if N == 0:
        return x, y
    lam = np.random.beta(1.0, 1.0)
    lam_fram = int(T * lam)
    lam_fram = lam_fram if lam_fram >= 1 else 1
    lam = lam_fram/T
    index = torch.randperm(N).to(x.device)
    x1 = resize_window(x, lam_fram)
    x2 = resize_window(x[index,:],T - lam_fram)
    mixed_x = torch.cat([x1, x2], dim=2)  # NCTV
    mixed_y = lam * y + (1 - lam) * y[index,:]
    return mixed_x, mixed_y


def Cutout_V(x,  y, max_ratio = 0.3):
    N,C,T,V,M = x.shape
    if N == 0:
        return x, y
    noise = torch.rand(V).to(x.device)
    ids_shuffle = torch.argsort(noise, dim=0)
    cut_num = int(random.random() * V * max_ratio)
    v_mask = torch.ones(V).to(x.device)
    v_mask[:cut_num] = 0.0
    v_mask = torch.gather(v_mask, dim=0, index=ids_shuffle)
    v_mask = v_mask[None, None, None, :, None]
    return v_mask*x, y


def Mix_V(x, y):
    N, S, C, T, V, M = x.shape
    if N == 0:
        return x, y
    index = torch.randperm(N).to(x.device)
    lam = 0
    mixed_x = torch.zeros_like(x).to(x.device)
    for i in range(V):
        if random.random() < 0.5:
           x_i = x[index].clone()
        else:
           lam += 1
           x_i = x.clone()
        x_i[:, :, :, :, i, :] = 0.0
        mixed_x += x_i
    lam = lam / V
    mixed_x = mixed_x/V
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def Mix_S(x, y, dim=3):
    N, C, T, V, M = x.shape
    if N == 0:
        return x, y
    index = torch.randperm(N).to(x.device)
    lam = 0
    x_index = x[index].clone()
    K = C // dim
    for i in range(K):
        if random.random() < 0.5:
            x[:, i * dim:(i + 1) * dim, :, :, :] = x_index[:, i * dim:(i + 1) * dim, :, :, :]
        else:
            lam += 1
    lam = lam / K
    y = lam * y + (1 - lam) * y[index]
    return x, y

def Mix_add(x, y, dim=3):
    lam = np.random.beta(2, 2)
    N, C, T, V, M = x.shape
    indices = np.random.permutation(N)
    x = x[:N] * lam + x[indices] * (1 - lam)
    target_shuffled_onehot = y[indices]
    y = y * lam + target_shuffled_onehot * (1 - lam)
    return x, y

def get_xyz(data):
    data_xyz = []
    N, C, T, V, M = data.shape
    for i in range(C):
        if type(data) is np.ndarray:
            data1 = data.copy()
        else:
            data1 = data.clone()
        data1[:, i] = 0.0
        data_xyz.append(data1)
    return data_xyz


def JDA(x, y, dim=3):
    N, C, T, V, M = x.shape
    xyz = get_xyz(x)
    x = torch.cat(xyz, dim=1)  # NCTVM
    theta = 0.5
    aug_list = [Mix_S, Mix_T]
    aug_order = np.random.permutation(len(aug_list))
    for i in range(len(aug_list)):
        aug_num = int(random.random() * theta * N)
        x[:aug_num], y[:aug_num] = aug_list[aug_order[i]](x[:aug_num], y[:aug_num])
        index = torch.randperm(N).to(x.device)
        x, y = x[index], y[index]
    N, C, T, V, M = x.shape
    x = x.view(N, C // dim, dim, T, V, M)
    x = x.sum(1) / (C // dim - 1)
    return x, y


def resize_window_feature(data, window):
    N, C, T, V= data.shape
    data = data.permute(0, 1, 3, 2).contiguous().view(N * C * V, T)
    data = data[None, None, :, :]
    data = torch.nn.functional.interpolate(data, size=(N * C * V, window), mode='bilinear',align_corners=False).squeeze()
    data = data.contiguous().view(N, C, V, window).permute(0, 1, 3, 2).contiguous()
    return data


def Mix_S_feature(x, y, dim=3):
    N, C, T, V = x.shape
    if N == 0:
        return x, y
    index = torch.randperm(N).to(x.device)
    lam = 0
    x_index = x[index].clone()
    K = C // dim
    for i in range(K):
        if random.random() < 0.5:
            x[:, i * dim:(i + 1) * dim, :, :] = x_index[:, i * dim:(i + 1) * dim, :, :]
        else:
            lam += 1
    lam = lam / K
    y = lam * y + (1 - lam) * y[index]
    return x, y


def Mix_T_feature(x, y):
    N, C, T, V = x.shape
    if N == 0:
        return x, y
    lam = np.random.beta(1.0, 1.0)
    lam_fram = int(T * lam)
    lam_fram = lam_fram if lam_fram >= 1 else 1
    lam = lam_fram/T
    index = torch.randperm(N).to(x.device)
    x1 = resize_window_feature(x, lam_fram)
    x2 = resize_window_feature(x[index,:],T - lam_fram)
    mixed_x = torch.cat([x1, x2], dim=2)  # NCTV
    mixed_y = lam * y + (1 - lam) * y[index,:]
    return mixed_x, mixed_y


def get_feature(data):
    data_feature = []
    N, C, T, V = data.shape
    data_feature.append(data)
    for i in range(C):
        if type(data) is np.ndarray:
            data1 = data.copy()
        else:
            data1 = data.clone()
        data1[:, i] = 0.0
        data_feature.append(data1)
    return data_feature


def JDA_add(x, y, dim=3):
    N, C, T, V, M = x.shape
    xyz = get_xyz(x)
    x = torch.cat(xyz, dim=1)  # NCTVM
    theta = 0.5
    aug_list = [Mix_S, Mix_T, Mix_add]
    aug_order = np.random.permutation(len(aug_list))
    for i in range(len(aug_list)):
        aug_num = int(random.random() * theta * N)
        x[:aug_num], y[:aug_num] = aug_list[aug_order[i]](x[:aug_num], y[:aug_num])
        index = torch.randperm(N).to(x.device)
        x, y = x[index], y[index]
    N, C, T, V, M = x.shape
    x = x.view(N, C // dim, dim, T, V, M)
    x = x.sum(1) / (C // dim - 1)
    return x, y


def JDA_add_plot(x, y, dim=3,file_name=None):
    N, C, T, V, M = x.shape
    xyz = get_xyz(x)
    x = torch.cat(xyz, dim=1)  # NCTVM
    theta = 0.5
    aug_list = [Mix_S, Mix_T, Mix_add]
    aug_order = np.random.permutation(len(aug_list))
    for i in range(len(aug_list)):
        aug_num = int(random.random() * theta * N)
        x[:aug_num], y[:aug_num] = aug_list[aug_order[i]](x[:aug_num], y[:aug_num])
        index = torch.randperm(N).to(x.device)
        x, y = x[index], y[index]
    N, C, T, V, M = x.shape
    x = x.view(N, C // dim, dim, T, V, M)
    x = x.sum(1) / (C // dim - 1)
    return x, y