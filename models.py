import torch, pickle
import numpy as np
from torch import nn
from definitions import Cube, _hash
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
        self.state3 = func.load('state3.pkl')

    def forward(self, cubes):
        inputs = self._make_inputs(cubes)
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
        sample_cubes = self._sample_cubes([x for x,m in zip(cubes, mask) if m], levels[mask], 15)
        _levels = self._predict_majority(sample_cubes, 30)
        levels[mask] = _levels
        if _levels.max() <= level:
            print(f'WARNING: model prediction is incorrect in first pass.')
        return levels

    def predict_multiple_pass(self, cubes):
        with torch.no_grad():
            levels = self.forward(cubes).argmax(-1).cpu().numpy()
            sample_cubes = self._sample_cubes(cubes, levels, 15)
            levels = self._predict_majority(sample_cubes, 30)
            for i,cube in enumerate(cubes):
                if cube.hash in self.state3:
                    levels[i] = 4 if cube.hash == Cube.FINALE else 3
            return levels

    def _sample_cubes(self, cubes, levels, samples):
        sample_operations = [
            [func.random_operations(x, np.random.randint(1, 10)) for _ in range(samples)]
            for x in levels]
        sample_cubes = [[x.copy() for _ in range(samples)] for x in cubes]
        for _sample_cubes, _sample_operations in zip(sample_cubes, sample_operations):
            for _cube, _operations in zip(_sample_cubes, _sample_operations):
                _cube.apply_operations(_operations)
        return sample_cubes

    def _predict_majority(self, sample_cubes, percentile):
        flatten_cubes = func.flatten(sample_cubes)
        with torch.no_grad():
            levels = self.forward(flatten_cubes).argmax(-1).view(len(sample_cubes), -1)
            levels = np.percentile(levels.cpu().numpy(), percentile, axis=-1)
            return levels.astype(np.int64)

    def _make_inputs(self, cubes):
        if torch.is_tensor(cubes):
            inputs = cubes
        else:
            inputs = np.array([x.numpy() for x in cubes])
            inputs = self.tensor(inputs).long()
        return inputs

    def tensor(self, values):
        device = next(self.parameters()).device
        return torch.tensor(values, device=device)

    def _hash(self, x):
        if torch.is_tensor(x):
            return _hash(x.cpu().numpy())
        else:
            return x.hash