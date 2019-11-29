import numpy as np
import torch
import torch.nn.functional as F
from transition import Transition
from replay_buffer import ReplayBuffer
import networks
from random import randint
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state_based_representation(observation, ghat, f1_model):

    if len(observation.shape) == 2:
        observation = np.expand_dims(observation, 0)

    observation = torch.Tensor(observation).to(DEVICE)

    start = time.time()
    Z_matrix = get_Z_matrix(f1_model, observation)

    # Check for batch
    if len(ghat.shape) == 1:
        ghat = ghat.unsqueeze(0)

    p_matrix = get_p_matrix(Z_matrix, ghat)
    zhat = get_zhat_matrix(observation, ghat, Z_matrix, p_matrix)
    return zhat

def get_Z_matrix(f1_model, observation):
    data = []
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                data.append(torch.cat((observation[i, j, :], observation[i, k, :]), 0).to(DEVICE))
    output = f1_model(torch.stack(data))
    return output.view(observation.shape[0], observation.shape[1], observation.shape[1], -1)

def get_p_matrix(Z_matrix, ghat):
    batch_size = len(Z_matrix)
    dim_1 = len(Z_matrix[0])
    w_matrix = torch.stack([torch.dot(z_vec, ghat[idx]) for idx, batch in enumerate(Z_matrix) for row in batch for z_vec in row])
    p_matrix = F.softmax(w_matrix.view(batch_size, -1), dim=1)
    return p_matrix.view(-1, dim_1, dim_1)

def get_zhat_matrix(observation, ghat, Z_matrix, p_matrix):
    z_vector = [[[0.0 for _ in range(5)] for _ in range(5)] for batch in observation]

    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                z_vector[i][j][k] = torch.sum(p_matrix[i][j][k] * Z_matrix[i][j][k])
    
    zhat = torch.stack([torch.stack([torch.sum(torch.stack(rows)) for rows in batch]) for batch in z_vector])

    state_rep = [[0.0 for _ in range(5)] for batch in observation]
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            current_o = observation[i, j, :]
            state_rep[i][j] = torch.cat([current_o, ghat[i], zhat[i]],0)

    return torch.stack([torch.stack(batch) for batch in state_rep])

def relabel_future_instructions(trajectory, t, k, discount_factor):
    t_size = len(trajectory)
    if t_size == t + 1:
        return []  # no future transitions
    delta_list = []

    for _ in range(k):
        future = randint(t+1, t_size-1)
        transition = trajectory[future]
        if len(transition.satisfied_goals_t) > 0:
            random_index = randint(0, len(transition.satisfied_goals_t)-1)
            goal_prime = transition.satisfied_goals_t[random_index]
            reward_prime = transition.reward * pow(discount_factor, future-t)
            delta_list.append([goal_prime, reward_prime])

    return delta_list
