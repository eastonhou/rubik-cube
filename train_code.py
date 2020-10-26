import torch, tqdm, random
import func
from definitions import Cube
from optimization import AutoDecay
from asynchronous import DataProducer

class Producer(DataProducer):
    def __init__(self):
        super(__class__, self).__init__(16, 65536)
        state3_codes = func.load('state3-codes.pkl')
        self.state3_codes = torch.tensor(tuple(state3_codes.values()), dtype=torch.uint8, device=0)
        code_steps = func.load('code-steps.pkl')
        codes = list(code_steps.keys())
        self.cache = {}
        self.codes = torch.tensor(codes, dtype=torch.uint8, device=0)
        self.xqueue = []
        self.reset()
        self.start()

    def _make_label(self, cube):
        if cube.hash in self.cache:
            return self.cache[cube.hash]
        _code = func.compute_relative_code(cube, Cube())
        _codes = self.state3_codes[:,_code]
        _codes2 = torch.cat((_codes, self.codes), dim=0)
        _, counts = _codes2.unique(return_counts=True, dim=0)
        self.cache[cube.hash] = result = counts.gt(1).sum() > 0
        return result

    def _produce(self):
        operations = func.random_operations(2, 36)
        cube = Cube()
        cube.apply_operations(operations)
        label = self._make_label(cube)
        self.xqueue.append((cube, label))
        self.put([(cube, label)])

    def print_summary(self):
        print(f'LABEL CORRECTION: {self.label_correction}')

    def reset(self):
        self.model = func.load_code_model().eval()
        self.label_correction = 0

    def get(self, size):
        cubes, labels = zip(*super(__class__, self).get(size, force_size=len(self.xqueue)<size))
        if len(cubes) < size:
            _records = random.choices(self.xqueue, k=size-len(cubes))
            _cubes, _labels = zip(*_records)
            cubes += _cubes
            labels += _labels
        return cubes, labels

class Trainer:
    def __init__(self):
        self.model = func.load_code_model()
        self.producer = Producer()

    def run(self, n=200, m=40, batch_size=256):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=8E-5)
        while True:
            self.producer.reset()
            loss = self.train_epoch(optimizer, n, batch_size)
            accuracy = self.evaluate_epoch(m, batch_size)
            admsg = f'ACCURACY={accuracy:>.6f} LOSS={loss:>.4F}'
            func.save_code_model(self.model)
            print(f'SAVE MODEL: {admsg}')
            self.producer.print_summary()

    def train_epoch(self, optimizer, n, batch_size):
        self.model.train()
        tloss = 0
        total = 0
        truth = 0
        timer = func.Timer()
        for _ in tqdm.tqdm(range(n)):
            cubes, labels = self.producer.get(batch_size)
            timer.check('generate-sample')
            logits = self.model(cubes)
            timer.check('forward')
            labels = self.model.tensor(labels).long()
            timer.check('tensor(labels)')
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
            tloss += loss.item()
            truth += labels.sum().item()
            total += batch_size
            loss.div_(labels.numel())
            timer.check('loss')
            optimizer.zero_grad()
            loss.backward()
            timer.check('backward')
            optimizer.step()
            timer.check('step')
        timer.print()
        print(f'TRUTH={truth} TOTAL={total} RATIO={truth/total:>.4F}')
        return tloss/total

    def evaluate_epoch(self, n, batch_size):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _ in tqdm.tqdm(range(n)):
                cubes, labels = self.producer.get(batch_size)
                logits = self.model(cubes)
                labels = self.model.tensor(labels)
                predicts = logits.argmax(-1)
                correct += predicts.eq(labels).sum().item()
                total += batch_size
        return correct/total

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
