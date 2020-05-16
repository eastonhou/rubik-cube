import torch
import numpy as np
from torch import nn
from definitions import Cube
import func

class Model(nn.Module):
    def __init__(self, embedding_dim=80):
        super(__class__, self).__init__()
        self.gridmap = self._create_gridmap()
        self.embeddings = nn.Embedding(len(self.gridmap), embedding_dim)
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
            nn.Linear(embedding_dim*3, 4))

    def forward(self, cubes):
        inputs = [[self.gridmap[g[0]] for g in func.flatten(x.data)] for x in cubes]
        inputs = self.tensor(inputs)
        x = self.embeddings(inputs)
        x = x.view(x.shape[0], -1)
        output = self.projection(x)
        return output

    def predict(self, cubes, samples=8):
        levels = self.forward(cubes).argmax(-1).cpu().numpy()
        operations = [
            [func.random_operations(x, np.random.randint(1, 10)) for _ in range(samples)]
            for x in levels]
        sample_cubes = [[x.copy() for _ in range(samples)] for x in cubes]
        for _sample_cubes, _sample_operations in zip(sample_cubes, operations):
            for _cube, _operations in zip(_sample_cubes, _sample_operations):
                func.apply_operations(_cube, _operations)
        flatten_cubes = func.flatten(sample_cubes)
        levels = self.forward(flatten_cubes).argmax(-1).view(-1, samples).cpu().numpy()
        least_levels = levels.min(axis=-1)
        return least_levels

    def tensor(self, values):
        device = next(self.parameters()).device
        return torch.tensor(values, device=device)

    def _create_gridmap(self):
        cube = Cube()
        gridmap = {}
        for face in cube.data:
            for grid in face:
                if grid[0] not in gridmap:
                    gridmap[grid[0]] = len(gridmap)
        return gridmap
