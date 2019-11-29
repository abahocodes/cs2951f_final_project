## Based on https://towardsdatascience.com/double-deep-q-networks-905dd8325412
## Please read to follow the code below, Double Q-learning in Hasselt et al., 2015 is used

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import argparse

import sys, os.path
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','clevr_robot_env'))
from clevr_robot_env import ClevrEnv
from networks import DQN, Encoder, OneHot
from replay_buffer import ReplayBuffer
from transition import Transition
from util import relabel_future_instructions
import time

import logging
logging._warn_preinit_stderr = 0
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="[test|train]")
parser.add_argument("--encoding", type=str, default="noncomp", help="[noncomp|onehot]")
parser.add_argument("--bins", type=int, default=1, help="bins in onehot encoding, eg: [1,4,10,20]")
args = parser.parse_args()

MODEL_FILE = 'agent-onehot-bin-'+str(args.bins)+'.npy'if args.encoding == "onehot" else 'agent-noncomp.npy'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", DEVICE)

REPLAY_BUFFER_SIZE = 2e6
BATCH_SIZE = 32
EPOCH = 200
EPISODES = 100
STEPS = 100
UPDATE_STEPS = 10

class DoubleDQN:
    def __init__(self, env, tau=0.1, gamma=0.9, epsilon=1.0):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.embedding_size = 30
        self.hidden_size = 30
        self.obs_shape = self.env.get_obs().shape
        self.action_shape = 40 // 5
        if args.encoding == "onehot":
            self.encoder = OneHot(args.bins, self.env.all_questions + self.env.held_out_questions, self.hidden_size).to(DEVICE)
        else:
            self.encoder = Encoder(self.embedding_size, self.hidden_size).to(DEVICE)

        self.model = DQN(self.obs_shape, self.action_shape, self.encoder).to(DEVICE)
        self.target_model = DQN(self.obs_shape, self.action_shape, self.encoder).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.epsilon = epsilon
        if os.path.exists(MODEL_FILE):
            checkpoint = torch.load(MODEL_FILE)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)


    def get_action(self, state, goal):
        assert len(state.shape) == 2 # This function should not be called during update

        if(np.random.rand() > self.epsilon):
            q_values = self.model.forward(state, goal)
            idx = torch.argmax(q_values).detach()
            obj_selection = idx // 8
            direction_selection = idx % 8
        else:
            action = self.env.sample_random_action()
            obj_selection = action[0]
            direction_selection = action[1]

        return int(obj_selection), int(direction_selection)

        
    def compute_loss(self, batch):     
        states, actions, goals, rewards, next_states, satisfied_goals, dones = batch

        rewards = torch.FloatTensor(rewards).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        curr_Q = self.model(states, goals) 

        curr_Q_prev_actions = [curr_Q[batch, actions[batch][0], actions[batch][1]] for batch in range(len(states))] # TODO: Use pytorch gather
        curr_Q_prev_actions = torch.stack(curr_Q_prev_actions)

        next_Q = self.target_model(next_states, goals) 
        
        next_Q_max_actions = torch.max(next_Q, -1).values
        next_Q_max_actions = torch.max(next_Q_max_actions, -1).values

        next_Q_max_actions = rewards + (1 - dones) * self.gamma * next_Q_max_actions

        loss = F.mse_loss(curr_Q_prev_actions, next_Q_max_actions.detach())

        return loss

    def update(self, replay_buffer, batch_size):
        for _ in range(UPDATE_STEPS):
            batch = replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_net(self): # TODO: Check this function
         # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(), 
            'target_model_state_dict': self.target_model.state_dict(), 
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
            }, MODEL_FILE)


def train(env, agent):
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for cycle in range(EPOCH):
        epoch_reward = 0.0
        agent.update_target_net() # target network update
        for episode in range(EPISODES):
            state = env.reset()
            goal, goal_program = env.sample_goal()
            env.set_goal(goal, goal_program)
            episode_reward = 0
            no_of_achieved_goals = 0
            current_instruction_steps = 0
            trajectory = []

            start = time.time()
            for step in range(STEPS):
                action = agent.get_action(state, goal)
                next_state, reward, done, _ = env.step(action, record_achieved_goal=True)
                achieved_goals = env.get_achieved_goals()
                transition = Transition(state, action, goal, reward, next_state, achieved_goals, done)
                trajectory.append(transition)
                episode_reward += reward

                if reward == 1.0:
                    goal, goal_program = env.sample_goal()
                    env.set_goal(goal, goal_program)
                    no_of_achieved_goals += 1
                    current_instruction_steps = 0

                if done:
                    break

                if current_instruction_steps == 10:
                    break
                
                current_instruction_steps += 1
                state = next_state
            # end = time.time()
            # print("environment interaction secs: ", end - start)
            # start = end
            
            for step in range(len(trajectory)):
                replay_buffer.add(trajectory[step])
                for goal_prime in trajectory[step].satisfied_goals_t:
                    transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, 1.0, trajectory[step].next_state, trajectory[step].satisfied_goals_t, trajectory[step].done)
                    replay_buffer.add(transition)
                deltas = relabel_future_instructions(trajectory, step, 4, 0.9)
                for delta in deltas:
                    goal_prime, reward_prime = delta
                    transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, reward_prime, trajectory[step].next_state, trajectory[step].satisfied_goals_t, trajectory[step].done)
                    replay_buffer.add(transition)    

            # end = time.time()
            # print("HIR secs: ", end - start)
            # start = end
            epoch_reward += episode_reward

            logging.error("[Episode] " + str(episode) + ": reward " + str(episode_reward) + " no of achieved goals: " + str(no_of_achieved_goals))

            agent.update(replay_buffer, BATCH_SIZE)   

            # end = time.time()
            # print("update model secs: ", end - start)
            # start = end


        logging.error("[Epoch] " + str(cycle) + ": total reward " + str(epoch_reward))
        agent.save_model()

        agent.epsilon *= 0.96
        if agent.epsilon < 0.1:
            agent.epsilon = 0.1


def test(env, agent):
    agent.epsilon = 0.0

    av_agent_steps = []
    av_random_steps = []

    with torch.no_grad():
        for _ in range(100):
            state = env.reset()
            number_of_steps_taken = 0
            goal, goal_program = env.sample_goal()
            env.set_goal(goal, goal_program)
            while True:
                action = agent.get_action(state, goal)
                next_state, reward, done, _ = env.step(action, record_achieved_goal=False)
                number_of_steps_taken += 1
                if reward == 1.0 or done:
                    break
                state = next_state
            av_agent_steps.append(number_of_steps_taken)

        for _ in range(100):
            state = env.reset()
            number_of_steps_taken = 0
            goal, goal_program = env.sample_goal()
            env.set_goal(goal, goal_program)
            while True:
                action = env.action_space.sample()
                obj_selection = action // 8
                direction_selection = action % 8
                next_state, reward, done, _ = env.step((obj_selection, direction_selection), record_achieved_goal=False)
                number_of_steps_taken += 1
                if reward == 1.0 or done:
                    break
                state = next_state
            av_random_steps.append(number_of_steps_taken)

    print("Agent Average Steps : " + str(sum(av_agent_steps)/len(av_agent_steps)))
    print("Random Average Steps: " + str(sum(av_random_steps)/len(av_random_steps)))

def main():
    env = ClevrEnv(action_type="perfect", obs_type='order_invariant', direct_obs=True, use_subset_instruction=True)
    agent = DoubleDQN(env)
    if args.mode == "train":
        train(env, agent)
    else:
        test(env, agent)

if __name__ == "__main__":
    main()
