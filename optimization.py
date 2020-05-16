class AutoDecay(object):
    def __init__(self, optimizer, patience=3, not_learning_counter=5, max_lr=1E-2, min_lr=1E-4):
        self.learning_rate = optimizer.defaults['lr']
        self.original_lr = self.learning_rate
        self.optimizer = optimizer
        self.counter = 0
        self.patience = patience
        self.not_learning_counter = not_learning_counter
        self.decay_counter = 0
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.last_accuracy = 0
        self.last_loss = 1E18

    def set_accuracy(self, accuracy):
        self.last_accuracy = accuracy

    def update_accuracy(self, accuracy, loss):
        if accuracy > self.last_accuracy or (accuracy == self.last_accuracy and loss < self.last_loss):
            self.last_accuracy = accuracy
            self.last_loss = loss
            self.better()
            return True
        else:
            self.worse()
            return False

    def better(self):
        self.counter = 0
        self.decay_counter = 0
        self.change_lr(self.learning_rate * 2)

    def worse(self):
        self.counter += 1
        if self.counter >= self.patience:
            self.change_lr(self.learning_rate / 2)
            self.counter = 0
            self.decay_counter += 1

    def change_lr(self, lr):
        self.learning_rate = min(max(lr, self.min_lr), self.max_lr)
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate

    def should_stop(self):
        return self.decay_counter >= self.not_learning_counter

    def reset(self):
        self.counter = 0
        self.decay_counter = 0
        self.last_accuracy = 0
        self.last_loss = 1E-18
        self.change_lr(self.original_lr)
