import torch
import torch.nn as nn
import random
import numpy as np
from scipy.special import softmax
from util import *
from time import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class OneHot(nn.Module):
    def __init__(self, bins, instructions, hidden_dim):
        super().__init__()
        self.bins = bins
        self.instruction_to_num = {instruction[0] : idx for idx, instruction in enumerate(instructions)}
        self.one_hot_sz = self.bins * len(self.instruction_to_num)
        self.output_sz = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(self.one_hot_sz, self.output_sz),
            nn.ReLU()
        ).to(DEVICE)

    def forward(self, q):
        if isinstance(q,np.ndarray):
            indexes = [self.instruction_to_num[x] * random.randint(0, self.bins - 1) for x in q] 
            one_hot = nn.functional.one_hot(torch.tensor(indexes), self.one_hot_sz)
        else: 
            one_hot = nn.functional.one_hot(torch.tensor(self.instruction_to_num[q] * random.randint(0, self.bins - 1)), self.one_hot_sz)
        return self.layers(one_hot.float())


class F1(nn.Module):
    def __init__(self, input_sz, output_sz):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_sz, output_sz),
            nn.ReLU()
        ).to(DEVICE)

    def forward(self, o):
        return self.layers(o)

class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab = None):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.vocab = self.get_vocab()
        self.output_sz = hidden_dim
        self.embedding = nn.Embedding(len(self.vocab), self.output_sz)

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
        return torch.LongTensor(ids).to(DEVICE)

    def forward(self, q):
        if isinstance(q,np.ndarray):
            return self._forward_batch(q)

        tokens = self.get_tokens(q)
        ids = self.tokens_to_id(tokens)

        embeddings = self.embedding(ids)
        outputs, _ = self.gru(embeddings.unsqueeze(1))

        return outputs[-1].squeeze(0)
    
    def _forward_batch(self, q): # Batch of questions
        
        tokens = [self.get_tokens(q[i]) for i in range(len(q))]

        ids = [self.tokens_to_id(tokens[i]) for i in range(len(q))]

        embeddings = [self.embedding(id_) for id_ in ids]
    
        outputs = [self.gru(embedings.unsqueeze(0))[0] for embedings in embeddings]
        outputs = [output[0][-1] for output in outputs]

        return torch.stack(outputs)



class DQN(nn.Module):
    def __init__(self, obs_shape, action_shape, encoder):
        super(DQN, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.encoder = encoder
        self.f1 = F1(self.obs_shape[1] * 2, encoder.output_sz).to(DEVICE)
        f3_input_shape = obs_shape[1] + encoder.output_sz + 5
        self.f3 = nn.Sequential(
            nn.Linear(f3_input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_shape)
        ).to(DEVICE)
        
    def forward(self, obs, g):
        start = time()
        g = self.encoder(g)
        print("encoder time: ", time() - start)
        start = time()
        zhat = get_state_based_representation(obs, g, self.f1)
        print("get state representation time: ", time() - start)
        return self.f3(zhat)
