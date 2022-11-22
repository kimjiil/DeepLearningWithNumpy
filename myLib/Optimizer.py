import numpy as np
from .Module import myTensor, myModule

class BaseOptimizer:
    def __init__(self, parameters, lr):
        # super(BaseOptimizer, self).__init__()
        self.params_iter = parameters
        self.learning_rate = lr
        self.step_t = 1

    def zero_grad(self):
        for param in self.params_iter:
            param.grad = None

    def step(self):
        # 여기서 parameter를 업데이트시킴

        for param in self.params_iter:
            param.update_parameter(param - self.learning_rate * param.grad)

    def add_param_group(self):
        ...

    def load_state_dict(self):
        ...

    def state_dict(self):
        ...

class Adam(BaseOptimizer):
    def __init__(self, parameters, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8):
        super(Adam, self).__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps


    def step(self):
        for param in self.params_iter:
            op = param._get_op()
            if not '_m' in param._opt_file and not '_v' in param._opt_file:
                param._opt_file['_m'] = op.zeros_like(param.grad)
                param._opt_file['_v'] = op.zeros_like(param.grad)

            param._opt_file['_m'] = self.beta1 * param._opt_file['_m'] + (1 - self.beta1) * param.grad
            param._opt_file['_v'] = self.beta2 * param._opt_file['_v'] + (1 - self.beta2) * (param.grad ** 2)

            biased_mt = param._opt_file['_m'] / (1 - self.beta1 ** self.step_t)
            biased_vt = param._opt_file['_v'] / (1 - self.beta2 ** self.step_t)
            test = self.learning_rate / (op.sqrt(biased_vt) + self.eps)
            _update = param - test  * biased_mt
            param.update_parameter(_update)
        self.step_t += 1

class Adagrad(BaseOptimizer):
    def __init__(self):
        super(Adagrad, self).__init__()

class SGD(BaseOptimizer):
    def __init__(self):
        super(SGD, self).__init__()