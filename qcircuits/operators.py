from qcircuits.tensors import Tensor
import numpy as np
from itertools import product


class Operator(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)
        # TODO check unitary

    def __repr__(self):
        return f'Operator for {self.rank // 2}-rank state space.'

    def __str__(self):
        s = f'Operator for {self.rank // 2}-rank state space. Tensor:\n'
        s += super().__str__()
        return s

    # Compose this operator with another operator, or apply it to a state vector
    # TODO break up this function
    def __call__(self, arg, qubit_indices=None):
        d = arg.rank
        if qubit_indices is not None:
            qubit_indices = list(qubit_indices)

        if qubit_indices is not None:
            if sorted(qubit_indices) != qubit_indices:
                raise NotImplementedError('Operator cannot be applied to indices out of order. ' \
                                          'Supplied qubit indices must be in ascending order.')
            if len(set(qubit_indices)) != len(qubit_indices):
                raise ValueError('Qubit indices list contains repeated elements.')

        # If we're applying to another operator, the ranks should match
        if type(arg) is Operator:
            op_indices = range(1, self.rank, 2)
            arg_indices = range(0, d, 2)

            if len(op_indices) != len(arg_indices):
                raise ValueError('An operator can only be composed with an operator of equal rank.')
            if qubit_indices is not None:
                raise ValueError('Qubit indices should only be supplied when applying an operator to a ' \
                                 'state vector, not composing it with another operator.')

        # We can apply an operator to a larger state, as long as we specify which axes of
        # the state vector are contracted (i.e., which qubits the operator is applied to).
        else:
            op_indices = range(1, self.rank, 2)
            arg_indices = range(d)

            if len(op_indices) > len(arg_indices):
                raise ValueError('An operator for a d-rank state space can only be applied to ' \
                                 'state vectors whose rank is >= d.')
            if len(op_indices) < len(arg_indices) and qubit_indices is None:
                raise ValueError('Applying operator to too-large state vector without supplying qubit indices.')
            if qubit_indices is not None:
                if min(qubit_indices) < 0:
                    raise ValueError('Supplied qubit index < 0.')
                if max(qubit_indices) >= len(arg_indices):
                    raise ValueError('Supplied qubit index larger than state vector rank.')

                if len(qubit_indices) == len(op_indices):
                    arg_indices = [arg_indices[index] for index in qubit_indices]
                else:
                    raise ValueError('Length of qubit_indices does not match number of operator ' \
                                     'lower indices (i.e., operator rank/2).')

        result = np.tensordot(self._t, arg._t, (op_indices, arg_indices))

        # Our convention is to have lower and upper indices of operators interleaved.
        # Using tensordot on operator-operator application leaves us with all upper
        # indices followed by all lower. We transpose the result to fix this.
        if type(arg) is Operator:
            permute = [0] * d
            permute[::2] = range(d//2)
            permute[1::2] = range(d//2, d)
            result = np.transpose(result, permute)
        # Likewise, application of operators to sub-vectors using tensordot leaves
        # our indices out of order, so we transpose them back.
        # This could be avoided with einsum, but it's easier to work with tensordot.
        elif qubit_indices is not None:
            permute = list(range(len(qubit_indices), d))
            for i, v in enumerate(qubit_indices):
                permute.insert(v, i)
            result = np.transpose(result, permute)

        return arg.__class__(result)


# Factory functions

# Identity operator
def Identity(d=1):
    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, 1.0 + 0.0j]])).tensor_power(d)

# Pauli X gate operator
# 'Not' gate, |0> to |1>, |1> to |0>
def PauliX(d=1):
    return Operator(np.array([[0.0j, 1.0 + 0.0j],
                              [1.0 + 0.0j, 0.0j]])).tensor_power(d)

# Pauli Y gate operator
def PauliY(d=1):
    return Operator(np.array([[0.0j, -1.0j],
                              [1.0j, 0.0j]])).tensor_power(d)

# Pauli Z gate operator
# Inverts the phase on |1>
def PauliZ(d=1):
    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, -1.0 + 0.0j]])).tensor_power(d)

# Hadamard gate operator
# |0> to |+>, |1> to |->
def Hadamard(d=1):
    return Operator(1/np.sqrt(2) *  np.array([[1.0 + 0.0j,  1.0 + 0.0j],
                                              [1.0 + 0.0j, -1.0 + 0.0j]])).tensor_power(d)

# Phase gate operator
# |0> to |0>, |1> to e^(i\phi) |1>
def Phase(phi=np.pi/2, d=1):
    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, np.exp(phi * 1j)]])).tensor_power(d)

# SqrtNot gate operator U^2 = X
def SqrtNot(d=1):
    return Operator(0.5 * np.array([[1 + 1j, 1 - 1j],
                                    [1 - 1j, 1 + 1j]])).tensor_power(d)

# Conditional-not gate operator
# Flips the second bit if the first bit is set
# |0x> to |0x>, |1x> to |1 (x+1)%2>
def CNOT():
    return Operator((1.0 + 0.0j) *  np.array([[[[ 1.0, 0.0],
                                                [ 0.0, 1.0]],
                                               [[ 0.0, 0.0],
                                                [ 0.0, 0.0]]],
                                              [[[ 0.0, 0.0],
                                                [ 0.0, 0.0]],
                                               [[ 0.0, 1.0],
                                                [ 1.0, 0.0]]]]))

# Swap gate operator
# |00> to |00>, |11> to |11>, |10> to |01>, |01> to |10>
def Swap():
    return Operator((1.0 + 0.0j) *  np.array([[[[ 1.0, 0.0],
                                                [ 0.0, 0.0]],
                                               [[ 0.0, 0.0],
                                                [ 1.0, 0.0]]],
                                              [[[ 0.0, 1.0],
                                                [ 0.0, 0.0]],
                                               [[ 0.0, 0.0],
                                                [ 0.0, 1.0]]]]))

# SqrtSwap gate operator, U^2 = Swap
def SqrtSwap():
    return Operator(np.array([[[[ 1.0,                 0.0],
                                [ 0.0,      0.5 * (1 + 1j)]],
                               [[ 0.0,                 0.0],
                                [ 0.5 * (1 - 1j),      0.0]]],
                              [[[ 0.0,       0.5 * (1 - 1j)],
                                [ 0.0,                 0.0]],
                               [[ 0.5 * (1 + 1j),      0.0],
                                [ 0.0,                 1.0]]]]))

# Conditional-U gate operator
# If the first bit is set, apply operator U to the remainder
def ControlledU(U):
    d = U.rank // 2 + 1
    shape = [2] * 2 * d
    t = np.zeros(shape, dtype=np.complex128)

    # If the first bit is zero, fill in as the identity operator.
    t[:, 0, ...] = Identity(d)[:, 0, ...]
    # Else, fill in as Identity tensored with U (Identity for the first bit,
    # which remains unchanged.
    t[:, 1, ...] = (Identity() * U)[:, 1, ...]
    return Operator(t)

# U_f operator
# If f is a Boolean function [0,1]^(d-1) -> [0, 1],
# constructs a Unitary operator that takes x, y to x, (y+f(x))%2
# There are d-1 input bits and 1 output bit.
def U_f(f, d):
    if d < 2:
        raise ValueError('U_f operator requires rank >= 2.')

    operator_shape = [2] * 2 * d
    t = np.zeros(operator_shape, dtype=np.complex128)

    for bits in product([0, 1], repeat=d):
        input_bits = bits[:-1]
        result = f(*input_bits)

        if result not in [0, 1]:
            raise RuntimeError('Function f for U_f operator should be Boolean,' \
                               'i.e., return 0 or 1.')

        result_bits = list(bits)
        if result:
            result_bits[-1] = 1 - result_bits[-1]
        all_bits = tuple([item for sublist in zip(result_bits, bits) for item in sublist])

        t[all_bits] = 1.0 + 0.0j

    return Operator(t)
