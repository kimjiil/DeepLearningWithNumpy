if __name__ == "__main__":
    from Module import myModule, myParameter
else:
    from .Module import myModule, myParameter

class MSELoss(myModule):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, label):
        self._backward_save = pred - label
        return 0.5 * self.op.mean(self.op.sum((pred - label) ** 2))

    def _backward(self, *args, **kwargs):
        return self._backward_save