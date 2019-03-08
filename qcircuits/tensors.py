import numpy as np

class Tensor:
    def __init__(self, tensor):
        self._t = tensor

    def __str__(self):
        return str(self._t)

    def __getitem__(self, key):
        return self._t[key]

    # TODO include the setter?
    #def __setitem__(self, key, value):
    #    self._t[key] = value

    @property
    def shape(self):
        return self._t.shape

    @property
    def rank(self):
        return len(self.shape)

    def product(self, arg):
        return self.__class__(np.tensordot(self._t, arg._t, axes=0))

    def power(self, n=2):
        t = self._t
        for i in range(n-1):
            t = np.tensordot(t, self._t, axes=0)

        return self.__class__(t)

    def __mul__(self, arg):
        return self.product(arg)

    def __pow__(self, n):
        return self.power(n)
