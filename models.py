import torch, pickle
import numpy as np
from torch import nn
from definitions import Cube
import func

class Model(nn.Module):
    def __init__(self, embedding_dim=80):
        super(__class__, self).__init__()
        self.embeddings = nn.Embedding(6, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim*54, embedding_dim*27),
            nn.BatchNorm1d(embedding_dim*27),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim*27, embedding_dim*13),
            nn.Tanh(),
            nn.Linear(embedding_dim*13, embedding_dim*6),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim*6, embedding_dim*3),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim*3, 5))

    def forward(self, cubes):
        inputs = np.array([x.numpy() for x in cubes])
        inputs = self.tensor(inputs).long()
        x = self.embeddings(inputs)
        x = x.view(x.shape[0], -1)
        output = self.projection(x)
        return output

    def predict(self, cubes, level, samples=8):
        with torch.no_grad():
            levels = self.forward(cubes).argmax(-1).cpu().numpy()
        if levels.max() <= level:
            return levels
        if level == 3:
            levels = np.array([4 if x.hash==Cube.FINALE else 3 for x in cubes])
            if levels.max() < 4:
                print(f'WARNING: model predicts a state3 cube as identity.')
            return levels
        mask = levels>level
        operations = [
            [func.random_operations(x, np.random.randint(1, 10)) for _ in range(samples)]
            for x in levels[mask]]
        sample_cubes = [[x.copy() for _ in range(samples)] for x,m in zip(cubes, mask) if m]
        for _sample_cubes, _sample_operations in zip(sample_cubes, operations):
            for _cube, _operations in zip(_sample_cubes, _sample_operations):
                _cube.apply_operations(_operations)
        flatten_cubes = func.flatten(sample_cubes)
        with torch.no_grad():
            _levels = self.forward(flatten_cubes).argmax(-1).view(-1, samples)
        least_levels = _levels.min(axis=-1)[0].cpu().numpy()
        levels[mask] = least_levels
        return levels

    def tensor(self, values):
        device = next(self.parameters()).device
        return torch.tensor(values, device=device)
