import torch
import torch.nn as nn
import random
import numpy as np
from scipy.special import softmax

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

# class f3(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(63,800)

#     def forward(self, z):
#         return self.linear(z)

class DQN(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(DQN, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.embedding_size = 64
        self.hidden_size = 64
        print(obs_shape)
        print(action_shape)
        self.f1 = f1(self.obs_shape[1] * 2, self.hidden_size)
        self.encoder = Encoder(self.embedding_size, self.hidden_size)
        f3_input_shape = obs_shape[0] * (obs_shape[1] + self.hidden_size + obs_shape[0])
        self.f3 = nn.Sequential(
            nn.Linear(f3_input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_shape)
        )
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, obs, g):
        g_tilde = self.encoder(g)
        O = np.asarray([[a, b] for a in obs for b in obs])
        fO = [self.f1(torch.Tensor(np.concatenate(o))) for o in O]
        Z = [torch.dot(t, g_tilde) for t in fO]
        p = torch.reshape(self.softmax(torch.stack(Z)), (obs.shape[0], -1))
        g = torch.stack([g_tilde] * obs.shape[0])
        o = torch.Tensor(obs)
        x = torch.cat((o, g, p), 1)
        return self.f3(x.reshape(-1))
    
    def act(self, state, goal):
        # state   = torch.Variable(torch.FloatTensor(state), volatile=True)
        q_value = self.forward(state, goal)
        action  = torch.argmax(q_value)
        return action
