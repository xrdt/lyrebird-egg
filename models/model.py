import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

strokes = np.load('../data/strokes.npy', encoding='bytes')

class Model(nn.Module):
    def __init__(self, hidden_size=121, n_output_mixture=20):
        super(Model, self).__init__()
        self.hidden_size = 121
        self.lstm = nn.LSTM(input_size=3, \
                            hidden_size=self.hidden_size, batch_first=True)

        # Prob of end of stroke
        self.e = nn.Linear(self.hidden_size, 1)
        # mixture weights
        self.pi = nn.Linear(self.hidden_size, n_output_mixture)
        # parameterize relative coord1 of stroke
        self.mu1 = nn.Linear(self.hidden_size, n_output_mixture)
        # parameterize relative coord2 of stroke
        self.mu2 = nn.Linear(self.hidden_size, n_output_mixture)
        # parameterize relative coord1 of stroke
        self.sigma1 = nn.Linear(self.hidden_size, n_output_mixture)
        # parameterize relative coord2 of stroke
        self.sigma2 = nn.Linear(self.hidden_size, n_output_mixture)
        # parameterize relative coords of stroke
        self.corr = nn.Linear(self.hidden_size, n_output_mixture)

    def forward(self, x, hidden):
        # run through hidden layers
        out, hidden = self.lstm(inputs, hiden)

        # run output transformations on all the lstm outputs
        e = nn.Sigmoid(self.e(out))
        pi = nn.Softmax(self.pi(out))
        mu1 = self.mu1(out)
        mu2 = self.mu2(out)
        sigma1 = torch.exp(self.sigma1(out))
        sigma2 = torch.exp(self.sigma2(out))
        corr = torch.tanh(self.corr(out))

        return e, pi, mu1, mu2, sigma1, sigma2, corr, hidden

def multivar_normal(x1, x2, mu1, mu2, sigma1, sigma2, corr):
    Z = ((x1-mu1)/sigma1)**2 + ((x2-mu2)/sigma2)**2 - \
        ((2*corr*(x1-mu1)*(x2-mu2))/(sigma1*sigma2))

    return torch.exp(-Z/(2*(1-corr**2))) / \
           (2*np.pi*sigma1*sigma2*torch.sqrt(1-corr**2))

def loss(e, pi, mu1, mu2, sigma1, sigma2, x1, x2, x3):
    loss = torch.sum(multivar_normal(x1, x2, mu1, mu2, sigma1, sigma2, corr) * pi)
    loss = -log(loss) - (x3 * log(e) + (1-x3)*log(1-e))

    return torch.mean(loss)
