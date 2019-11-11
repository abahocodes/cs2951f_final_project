import numpy as np
import torch
from transition import Transition
from replay_buffer import ReplayBuffer
import networks
from random import randint

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

def relabel_future_instructions(trajectory, t, k, discount_factor):
    delta_list = []

    t_size = len(trajectory)

    for i in range(k):
        future = randint(t+1, t_size-1)
        transition = trajectory[future]
        if len(transition.satisfied_goals_t) > 0:
            random_index = randint(0, len(transition.satisfied_goals_t)-1)
            goal_prime = transition.satisfied_goals_t[random_index]
            reward_prime = transition.reward * pow(discount_factor, future-t)
            delta_list.append([goal_prime, reward_prime])
    
    return delta_list

        

def hindsight_instruction_replay(language_supervisor, environment, k):
    buffer_size = 0 # TBD
    episode_count = 30 # M value
    t_count = 30 # Capital T value
    discount_factor = 30

    buffer = ReplayBuffer(buffer_size)
    dqn_policy = DQN() # TODO: Add parameters to DQN

    for i in range(episode_count):
        start_state = None # TODO: Returned by environment
        instruction_goal = None # TODO: Returned by language supervisor
        
        trajectory = []

        # t_count size lists
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        # next_action_list = [] # not used
        # unsatisfied_goals_list = [] # U_t - not used
        satisfied_goals_list = []  # V_t
        
        state_t = start_state
    
        for t in range(t_count):
            #unsatisfied_goals_t = [] # TODO: Returned by using language supervisor and current state
            
            action_t = dqn_policy.act(state_t, instruction_goal)
            next_state_t = None # take action_t from current state

            reward_t = None # TODO: Returned by reward function which takes the current state and goal as parameters
            satisfied_goals_t = set() # TODO: unsatisfied_goals - currently_unsatisfied goals

            transition = Transition(state_t, action_t, instruction_goal, reward_t, next_state_t, satisfied_goals_t)
            trajectory.append(transition)

            if reward_t == 1:
                instruction_goal = None # Get another goal from language supervisor

            state_list.append(state_t)
            action_list.append(action_t)
            reward_list.append(reward_list)
            # unsatisfied_goals_list.append(unsatisfied_goals_t)
            satisfied_goals_list.append(satisfied_goals_t)

            state_t = next_state_t
        
        for t in range(t_count):
            transition = Transition(
                state_list[t], 
                action_list[t], 
                instruction_goal, 
                reward_list[t], 
                next_state_list[t], 
                satisfied_goals_list[t])
            buffer.add(transition)

            for goal_prime in satisfied_goals_list[t]:
                transition = Transition(
                    state_list[t], 
                    action_list[t], 
                    goal_prime, 
                    reward_list[t], 
                    next_state_list[t], 
                    satisfied_goals_list[t])
                buffer.add(transition)
            
            delta_list = relabel_future_instructions(trajectory, t, k, discount_factor)
            for delta in delta_list:
                goal_prime, reward_prime = delta
                transition = Transition(
                    state_list[t], 
                    action_list[t], 
                    goal_prime, 
                    reward_prime, 
                    next_state_list[t], 
                    satisfied_goals_list[t])
                buffer.add(transition)    
        #TODO: Update DQN 
        return dqn_policy          






