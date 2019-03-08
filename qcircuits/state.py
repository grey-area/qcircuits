from qcircuits.tensors import Tensor
import qcircuits.operators as operators
import numpy as np


class State(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)

        if abs(np.sum(self.probabilities) - 1.0) > 1e-4:
            raise RuntimeError('Vector is not a unit vector.')

    def __repr__(self):
        return f'State vector for {self.rank}-rank state space.'

    def __str__(self):
        s = f'State vector for {self.rank}-rank state space. Tensor:\n'
        s += super().__str__()
        return s

    def dot(self, arg):
        return np.sum(np.conj(self._t) * arg._t)

    @property
    def probabilities(self):
        probs = np.real(np.conj(self._t) * self._t)
        assert ( abs(np.sum(probs) - 1.0) < 1e-4 ), 'State probabilities do not sum to 1.'
        return probs

    @property
    def amplitudes(self):
        amp = np.copy(self._t)
        amp.flags.writeable = False
        return amp

    # TODO measure wrt more than 1 qubit?
    # TODO option to reduce state vector or not
    def measure(self, qubit_index):
        if qubit_index < 0 or qubit_index >= self.rank:
            raise ValueError('Trying to measure qubit whose index i is not 0<=i<d, ' \
                             'where d is the rank of the state vector.')

        state = np.swapaxes(self._t, 0, qubit_index)
        ps = np.reshape(np.real(state * np.conj(state)), (2, -1)).sum(axis=1)
        bit = np.random.choice([0, 1], p=ps)
        p = ps[bit]
        state = state[bit, ...] / np.sqrt(p)
        return bit, State(state)


# Factory functions

def qubit(*, alpha=None, beta=None,
          global_phase=0.0, theta=None, phi=None):
    if all([v is not None for v in [alpha, beta]]) and all([v is None for v in [theta, phi]]):
        tensor = np.zeros(2, dtype=np.complex128)
        tensor[0] = alpha
        tensor[1] = beta
    elif all([v is not None for v in [theta, phi]]) and all([v is None for v in [alpha, beta]]):
        tensor = np.zeros(2, dtype=np.complex128)
        alpha = np.cos(theta/2)
        beta = np.sin(theta/2) * np.exp(1j * phi)
        tensor[0] = alpha
        tensor[1] = beta
    else:
        raise ValueError('Supplied incorrect combination of arguments for qubit. '\
            'Supply alpha and beta, or theta and phi.')

    return State(tensor * np.exp(1j * global_phase))

def zeros(d=1):
    if d < 1:
        raise ValueError('Rank must be at least 1.')

    shape = [2] * d
    t = np.zeros(shape, dtype=np.complex128)
    t.flat[0] = 1
    return State(t)

def ones(d=1):
    if d < 1:
        raise ValueError('Rank must be at least 1.')

    shape = [2] * d
    t = np.zeros(shape, dtype=np.complex128)
    t.flat[-1] = 1
    return State(t)

# Construct a state vector in one of the computational basis states
# from a bitstring
# TODO rename?
def bitstring(*bits):
    if len(bits) < 1:
        raise ValueError('Rank must be at least 1.')

    if bits[0]:
        q = ones()
    else:
        q = zeros()
    for bit in bits[1:]:
        if bit:
            q = q * ones()
        else:
            q = q * zeros()
    return q

def positive_superposition(d=1):
    if d < 1:
        raise ValueError('Rank must be at least 1.')

    H = operators.H(d)
    x = zeros(d)
    return H(x)

def bell_state(a=0, b=0):
    if a not in [0, 1] or b not in [0, 1]:
        raise ValueError('Bell state arguments are bits, and must be 0 or 1.')

    phi = bitstring(a, b)
    phi = operators.H()(phi, qubit_indices=[0])

    return operators.CNOT()(phi)
