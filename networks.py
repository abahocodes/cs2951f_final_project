import torch.nn as nn
import random

class f1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = nn.Linear(24,1)
    def forward(self, o):
        return self.linear(o)

class f3(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = nn.Linear(63,800)
    def forward(self, z):
        return self.linear(z)

class DQN(nn.Module):
    def __init__(self, env):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(63, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, instruction):
        state   = torch.Variable(torch.FloatTensor(state), volatile=True)
        q_value = self.forward(state)
        action  = torch.argmax(q_value)
        return action

class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab = None):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.vocab = self.get_vocab()
        self.embedding = nn.Embedding(len(vocab), hidden_dim)

    def get_vocab(self):
        vocab_words = open('assets/vocab.txt','r').read().split('\n')
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
        ids = self.token_to_id(tokens)

        embeddings = self.embedding(ids)
        outputs, _ = self.gru(embeddings.unsqueeze(1))

        return outputs[-1].squeeze(0)