import numpy as np
import torch

def get_state_based_representation(observation, instruction):
    Z_matrix = get_Z_matrix(f1, observation)
    ghat = f2(instruction)
    w_matrix = get_w_matrix(Z_matrix, ghat)
    p_matrix = get_p_matrix(w_matrix)
    zhat = get_zhat_matrix(observation, ghat, Z_matrix, p_matrix)
    return zhat

def get_Z_matrix(f1_model, observation):
    assert observation.shape[0] == 5
    Z_matrix = [[1 for _ in range(5)] for _ in range(5)]

    for i, o1 in enumerate(observation):
        for j, o2 in enumerate(observation):
            o1_ = torch.from_numpy(o1).float()
            o2_ = torch.from_numpy(o2).float()
            pair = torch.cat((o1_, o2_), 0)
            value = f1_model(pair)
            Z_matrix[i][j] = value

    return Z_matrix

def get_w_matrix(Z_matrix, ghat):
    w_matrix = [[1 for _ in range(5)] for _ in range(5)]
    for i, _ in enumerate(Z_matrix):
        for j, _ in enumerate(Z_matrix):
            w_matrix[i][j] = torch.sum(ghat*Z_matrix[i][j])
    return w_matrix

def softmax(w_matrix):
    assert len(w_matrix) == len(w_matrix[0])
    max_val = max([max(w_matrix[i]) for i in range(len(w_matrix))])
    denom = 0
    new_matrix = [[1 for _ in range(len(w_matrix))] for _ in range(len(w_matrix))]

    for i, _ in enumerate(w_matrix):
        for j, _ in enumerate(w_matrix):
            denom += torch.exp(w_matrix[i][j] - max_val)

    for i, _ in enumerate(w_matrix):
        for j, _ in enumerate(w_matrix):
            new_matrix[i][j] = torch.exp(w_matrix[i][j] - max_val) / denom

    return new_matrix
    

def get_p_matrix(w_matrix):
    new_matrix = softmax(w_matrix)
    return new_matrix

def get_zhat_matrix(observation, ghat, Z_matrix, p_matrix):
    assert len(Z_matrix) == len(p_matrix)
    assert len(observation) == len(Z_matrix)

    z_value = 0
    for i, _ in enumerate(p_matrix):
        for j, _ in enumerate(p_matrix):
            z_value += (p_matrix[i][j] * Z_matrix[i][j])

    new_matrix = []
    for o in observation:
        row = torch.cat([torch.from_numpy(o).float(), ghat, zvalue],0)
        new_matrix.append(row)

    return torch.stack(new_matrix, 0)