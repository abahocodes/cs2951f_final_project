import torch.nn as nn
import random
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
        ids = self.token_to_id(tokens)

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
        f3_input_shape = obs_shape[0] * (obs_shape[1] + self.hidden_size + 1)
        self.f3 = nn.Sequential(
            nn.Linear(f3_input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_shape)
        )
        
    def forward(self, obs, g):
        g_tilde = self.encoder(g)
        O = [(a, b) for a in obs for b in obs]
        O = np.apply_along_axis(lambda o : self.f1(np.concatenate(o)), O).reshape(self.obs_shape[0], self.obs_shape[0], -1)
        Z = np.sum(g_tilde * O)
        p = softmax(Z)
        print("p matrix shape: ", p.shape)
        print("O matrix shape: ", O.shape)
        return self.layers(x)
    
    def act(self, state, instruction):
        state   = torch.Variable(torch.FloatTensor(state), volatile=True)
        q_value = self.forward(state)
        action  = torch.argmax(q_value)
        return action
