if __name__ == "__main__":
    from Module import myModule, myParameter
else:
    from .Module import myModule, myParameter

class testloss(myModule):
    def __init__(self):
        super(testloss, self).__init__()

    def forward(self, pred, label):
        return 0.5 * self.op.mean(self.op.sum((pred - label) ** 2))

    def backward(self, *args, **kwargs):
        # print("Loss backward!!!")
        self.backward_fn(*args, **kwargs)