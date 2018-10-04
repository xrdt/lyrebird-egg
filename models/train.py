import torch
from model import *
import numpy as np
import sys
sys.path.append('../')
from utils.dataloader import DataLoader

# Train the unconditional model
def train(epochs=10, batch_size=50, lr=1e-3):
    dataloader = DataLoader('/data/strokes.npy')
    Model = model.Model()
    optimizer = torch.optim.RMSprop(Model.params(), lr=lr)

    for epoch in range(epochs):
        x, y = dataloader.generate_batch()
        x = torch.from_numpy(np.array(x))
        y = torch.from_numpy(np.array(y))

        y1 = y[:, :, 0]
        y2 = y[:, :, 1]
        y2 = y[:, :, 2]

        hidden = autograd.Variable(torch.randn(1, x.size(0), 121))

        e, pi, mu1, mu2, sigma1, sigma2, corr, hidden = Model(x, hidden)

        loss_val = loss(e, pi, mu1, mu2, sigma1, sigma2, corr, y1, y2, y3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(Model.save_dict(), 'unconditional.pt')

'''def generate_unconditionally():
    network =
    network.load_state_dict()

    x =



def generate_conditionally():
'''
