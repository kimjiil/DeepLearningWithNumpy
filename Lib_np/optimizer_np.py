import numpy as np

class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8):
        self.step = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps

        self._m = dict()
        self._v = dict()


    def update(self, layers):
        if len(self._m) == 0 or len(self._v) == 0:
            self._init_mv(layers)

        for idx, layer in enumerate(layers):
            gradient = layer.get_weight
            if gradient() is None:
                continue
            dw, w, db, b = gradient()
            dw_key = f"dw{idx}"
            db_key = f"db{idx}"
            self._m[dw_key] = self.beta1 * self._m[dw_key] + (1 - self.beta1) * dw
            self._m[db_key] = self.beta1 * self._m[db_key] + (1 - self.beta1) * db

            self._v[dw_key] = self.beta2 * self._v[dw_key] + (1 - self.beta2) * (dw ** 2)
            self._v[db_key] = self.beta2 * self._v[db_key] + (1 - self.beta2) * (db ** 2)

            bias_correction1_w = self._m[dw_key] / (1 - (self.beta1 ** self.step))
            bias_correction1_b = self._m[db_key] / (1 - (self.beta1 ** self.step))

            bias_correction2_w = self._v[dw_key] / (1 - (self.beta2 ** self.step))
            bias_correction2_b = self._v[db_key] / (1 - (self.beta2 ** self.step))

            next_w = w - self.lr * bias_correction1_w / (np.sqrt(bias_correction2_w) + self.eps)
            next_b = b - self.lr * bias_correction1_b / (np.sqrt(bias_correction2_b) + self.eps)

            layer.set_weight(w=next_w, b=next_b)


    def _init_mv(self, layers):
        for idx, layer in enumerate(layers):

            gradient = layer.get_weight
            if gradient() is None:
                continue

            dw, w, db, b = gradient()
            dw_key = f"dw{idx}"
            db_key = f"db{idx}"

            self._m[dw_key] = np.zeros_like(dw)
            self._m[db_key] = np.zeros_like(db)

            self._v[dw_key] = np.zeros_like(dw)
            self._v[db_key] = np.zeros_like(db)



