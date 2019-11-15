import torch
import torch.nn as nn
import random
import numpy as np
from scipy.special import softmax
from util import *

class f1(nn.Module):
    def __init__(self, input_sz, output_sz):
        super().__init__()
        self.linear = nn.Linear(input_sz, output_sz)

    def forward(self, o):
        return self.linear(o)

class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab = None):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.vocab = self.get_vocab()
        self.embedding = nn.Embedding(len(self.vocab), hidden_dim)

    def get_vocab(self):
        vocab_words = open('../clevr_robot_env/assets/vocab.txt','r').read().split('\n')
        vocab_size = len(vocab_words)
        vocab = dict(zip(vocab_words, range(vocab_size)))
        return vocab

    def purify(self, text):
        return text.replace(',',' ,').replace(';',' ;').replace('?',' ?')

    def get_tokens(self, text):
        text = self.purify(text)
        return text.split()

    def tokens_to_id(self, tokens):
        ids = [self.vocab[t.lower()] for t in tokens]
        return torch.LongTensor(ids)

    def forward(self, q):
        tokens = self.get_tokens(q)
        ids = self.tokens_to_id(tokens)

        embeddings = self.embedding(ids)
        outputs, _ = self.gru(embeddings.unsqueeze(1))

        return outputs[-1].squeeze(0)

class DQN(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(DQN, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.embedding_size = 50
        self.hidden_size = 50
        self.f1 = f1(self.obs_shape[1] * 2, self.hidden_size)
        self.encoder = Encoder(self.embedding_size, self.hidden_size)
        f3_input_shape = obs_shape[1] + self.hidden_size + 5
        self.f3 = nn.Sequential(
            nn.Linear(f3_input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_shape)
        )
        
    def forward(self, obs, g):
        zhat = get_state_based_representation(obs, g, self.f1, self.encoder)
        return self.f3(zhat)
    
    def act(self, state, goal):
        q_value = self.forward(state, goal)
        action  = torch.argmax(q_value)
        return action
