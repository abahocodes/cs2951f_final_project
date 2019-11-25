import numpy as np
import torch
from transition import Transition
from replay_buffer import ReplayBuffer
import networks
from random import randint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state_based_representation(observation, instruction, f1_model, f2):

    if len(observation.shape) == 2:
        observation = np.expand_dims(observation, 0)

    observation = torch.Tensor(observation).to(DEVICE)
    Z_matrix = get_Z_matrix(f1_model, observation)
    ghat = f2(instruction)

    # Check for batch
    if len(ghat.shape) == 1:
        ghat = ghat.unsqueeze(0)

    w_matrix = get_w_matrix(Z_matrix, ghat)
    p_matrix = get_p_matrix(w_matrix)
    zhat = get_zhat_matrix(observation, ghat, Z_matrix, p_matrix)
    return zhat

def get_Z_matrix(f1_model, observation):
    Z_matrix = [[[0.0 for _ in range(5)] for _ in range(5)] for batch in observation]

    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                pair = torch.cat((observation[i, j, :], observation[i, k, :]), 0)
                pair = pair.to(DEVICE)
                value = f1_model(pair)
                Z_matrix[i][j][k] = value

    return Z_matrix

def get_w_matrix(Z_matrix, ghat):
    w_matrix = [[[0.0 for _ in range(5)] for _ in range(5)] for batch in Z_matrix]
    for i in range(len(w_matrix)):
        for j in range(len(w_matrix[0])):
            for k in range(len(w_matrix[0][0])):
                w_matrix[i][j][k] = Z_matrix[i][j][k] @ ghat[i]
    return w_matrix

def softmax(w_matrix):
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
    new_matrix = [softmax(batch) for batch in w_matrix]
    return new_matrix

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
