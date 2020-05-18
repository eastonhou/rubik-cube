import torch, tqdm
import numpy as np
import func
from optimization import AutoDecay
from asynchronous import DataProducer

class Producer(DataProducer):
    def __init__(self):
        super(__class__, self).__init__(16, 65536)
        self.reset()
        self.start()

    def _produce(self):
        if np.random.ranf() < 0.998:
            label = np.random.randint(4)
        else:
            label = 4
        cube = func.make_cube(label)
        self.put([(cube, label)])

    def print_summary(self):
        print(f'LABEL CORRECTION: {self.label_correction}')

    def reset(self):
        self.model = func.load_model().eval()
        self.label_correction = 0

    def get(self, size):
        cubes, labels0 = zip(*super(__class__, self).get(size, force_size=True))
        labels = self.model.predict_multiple_pass(cubes)
        self.label_correction += (labels != labels0).sum()
        return cubes, labels

class Trainer:
    def __init__(self):
        self.model = func.load_model()
        self.producer = Producer()

    def run(self, n=2000, m=1000, batch_size=256):
        optimizer = torch.optim.Adam(self.model.parameters())
        accuracy = self.evaluate_epoch(m, batch_size)
        print(f'INITIAL ACCURACY={accuracy:>.4f}')
        autodecay = AutoDecay(optimizer, max_lr=2E-3, min_lr=2E-5)
        autodecay.update_accuracy(accuracy, 1000)
        while True:
            self.producer.reset()
            loss = self.train_epoch(optimizer, n, batch_size)
            accuracy = self.evaluate_epoch(m, batch_size)
            admsg = f'ACCURACY={accuracy:>.4f}/{autodecay.last_accuracy:>.4F}'\
                f' LOSS={loss:>.4F}'\
                f' C/D={autodecay.counter}/{autodecay.decay_counter}'\
                f' lr={autodecay.learning_rate}'
            if autodecay.update_accuracy(accuracy, loss):
                func.save_model(self.model)
                print(f'SAVE MODEL: {admsg}')
            else:
                print(f'CONTINUE: {admsg}')
            if autodecay.should_stop():
                self.model = func.load_model()
                autodecay.reset()
            self.producer.print_summary()

    def train_epoch(self, optimizer, n, batch_size):
        self.model.train()
        tloss = 0
        total = 0
        timer = func.Timer()
        for _ in tqdm.tqdm(range(n)):
            cubes, labels = self.producer.get(batch_size)
            timer.check('generate-sample')
            logits = self.model(cubes)
            timer.check('forward')
            labels = self.model.tensor(labels)
            timer.check('tensor(labels)')
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
            tloss += loss.item()
            total += batch_size
            loss.div_(labels.numel())
            timer.check('loss')
            optimizer.zero_grad()
            loss.backward()
            timer.check('backward')
            optimizer.step()
            timer.check('step')
        timer.print()
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
