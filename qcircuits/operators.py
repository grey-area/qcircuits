"""
The operators module contains the Operator class, instances of which
represent Operators on vector spaces for multi-qubit systems, and
factory functions for creating specific operators.

Each of the factory functions (but not the Operator class) is aliased
at the top-level module, so that, for example, one can call
``qcircuits.Hadamard()`` instead of ``qcircuits.state.Hadamard()``.
"""


from itertools import product
import copy

import numpy as np

from qcircuits.tensors import Tensor


class Operator(Tensor):
    """
    A container class for a tensor representing an operator on a vector
    space for a quantum system, and associated methods.

    Parameters
    ----------
    tensor : numpy complex128 multidimensional array
        The tensor representing the operator.
    """

    def __init__(self, tensor):
        super().__init__(tensor)
        # TODO check unitary (maybe only check when applying?)

    @staticmethod
    def from_matrix(M):
        """
        QCircuits represents operators for d-qubit systems with type (d, d) tensors.
        This function constructs an operator from the more common matrix
        Kronecker-product representation of the operator.

        Parameters
        ----------
        numpy complex128 multidimensional array
            The matrix representation of the operator.

        Returns
        -------
        Operator
            A d-qubit operator.

        """

        if type(M) is list:
            M = np.array(M, dtype=np.complex128)

        # Check the matrix is square
        shape = M.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('The matrix should be square.')
        # Check the dimension is a power of 2
        d = np.log2(shape[0])
        if not d.is_integer():
            raise ValueError('The matrix dimension should be a power of 2.')
        d = int(d)

        permutation = [0] * 2 * d
        permutation[::2] = range(0, d)
        permutation[1::2] = range(d, 2*d)

        return Operator(M.reshape([2] * 2 * d).transpose(permutation))

    def __repr__(self):
        s = 'Operator('
        s += super().__str__().replace('\n', '\n' + ' ' * len(s))
        s += ')'
        return s

    def __str__(self):
        s = 'Operator for {}-qubit state space.'.format(self.rank // 2)
        s += ' Tensor:\n'
        s += super().__str__()
        return s

    @property
    def adj(self):
        """
        Get the adjoint/inverse of this operator,
        :math:`A^{\dagger} = (A^{*})^{T}`. As the operator is unitary,
        :math:`A A^{\dagger} = I`.

        Returns
        -------
        Operator
            The adjoint operator.
        """

        d = self.rank
        permutation = [0] * d
        permutation[::2] = range(1, d, 2)
        permutation[1::2] = range(0, d, 2)
        t = np.conj(self._t).transpose(permutation)
        return Operator(t)

    def to_matrix(self):
        """
        QCircuits represents operators for d-qubit systems with type (d, d) tensors.
        This function returns the more common matrix Kronecker-product
        representation of the operator.

        Returns
        -------
        numpy complex128 multidimensional array
            The matrix representation of the operator.
        """

        d = len(self.shape) // 2
        permutation = list(range(0, 2*d, 2)) + list(range(1, 2*d, 2))
        return self._t.transpose(permutation).reshape(2**d, 2**d)

    def _permuted_tensor(self, axes, inverse=False):
        if inverse:
            axes = np.argsort(axes)

        op_axes = [[2*n, 2*n+1] for n in axes]
        op_axes = [v for sublist in op_axes for v in sublist]

        return np.transpose(self._t, op_axes)


    def permute_qubits(self, axes, inverse=False):
        """
        Permute the qubits (i.e., both the incoming and outgoing wires)
        of the operator.

        Parameters
        ----------
        axes : list of int
            Permute the qubits according to the values given.
        inverse : bool
            If true, perform the inverse permutation of the qubits.
        """

        self._t = self._permuted_tensor(axes, inverse=inverse)


    def swap_qubits(self, axis1, axis2):
        """
        Swap two qubits (i.e., both the incoming and outgoing wires)
        of the operator.

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.
        """

        self._t = np.swapaxes(self._t, 2*axis1, 2*axis2)
        self._t = np.swapaxes(self._t, 2*axis1 + 1, 2*axis2 + 1)

    def __add__(self, arg):
        return Operator(self._t + arg._t)

    def __sub__(self, arg):
        return self + (-1) * arg

    def __mul__(self, scalar):
        if isinstance(scalar, (float, int, complex)):
            return Operator(scalar * self._t)
        else:
            return super().__mul__(scalar)

    def __rmul__(self, scalar):
        if isinstance(scalar, (float, int, complex)):
            return Operator(scalar * self._t)

    def __truediv__(self, scalar):
        return Operator(self._t / scalar)

    def __neg__(self):
        return Operator(-self._t)

    def __call__(self, arg, qubit_indices=None):
        """
        Applies this Operator to another Operator, as in operator
        composition A(B), or to a :py:class:`.State`, as in A(v). This means that
        if two operators A and B will be applied to state v in sequence,
        either B(A(v)) or (B(A))(v) are valid.

        A d-qubit operator may be applied to an n-qubit system with :math:`n>d`
        if the qubits to which it is to be applied are specified in the
        `qubit_indices` parameter.

        Parameters
        ----------
        arg : State or Operator
            The state that the operator is applied to, or the operator
            with which the operator is composed.
        qubit_indices: list of int
            If the operator is applied to a larger
            quantum system, the user must supply a list of the indices
            of the qubits to which the operator is to be applied.
            These can also be used to apply the operator to the qubits
            in arbitrary order.

        Returns
        -------
        State or Operator
            The state vector or operator resulting in applying the
            operator to the argument.
        """

        if type(arg) is Operator:
            d = arg.rank // 2
            arg_indices = list(range(0, 2*d, 2))
        else:
            d = arg.rank
            arg_indices = list(range(d))
        op_indices = list(range(1, self.rank, 2))

        if len(op_indices) > d:
            raise ValueError('An operator for a d-rank state space can only be applied to '
                             'a system whose rank is >= d.')
        if len(op_indices) < d and qubit_indices is None:
            raise ValueError('Applying operator to too-large system without supplying '
                             'qubit indices.')

        if qubit_indices is not None:
            qubit_indices = list(qubit_indices)

            if len(set(qubit_indices)) != len(qubit_indices):
                raise ValueError('Qubit indices list contains repeated elements.')
            if min(qubit_indices) < 0:
                raise ValueError('Supplied qubit index < 0.')
            if max(qubit_indices) >= d:
                raise ValueError('Supplied qubit index larger than system size.')
            if len(qubit_indices) != len(op_indices):
                raise ValueError('Length of qubit_indices does not match operator.')
        else:
            qubit_indices = list(range(d))

        d1 = len(qubit_indices)
        non_application_indices = sorted(list(set(range(d))  - set(qubit_indices)))
        application_permutation = qubit_indices + non_application_indices

        arg_t = arg._permuted_tensor(application_permutation)
        arg_indices = arg_indices[:len(qubit_indices)]
        result = np.tensordot(self._t, arg_t, (op_indices, arg_indices))

        # Our convention is to have lower and upper indices of operators interleaved.
        # Using tensordot on operator-operator application leaves us with all upper
        # indices followed by all lower. We transpose the result to fix this.
        if type(arg) is Operator:
            permute = [0] * 2*d
            permute[: 2*d1 : 2] = range(d1)
            permute[1 : 2*d1 : 2] = range(d1, 2*d1)
            permute[2*d1 : 2*d] = range(2*d1, 2*d)
            result = np.transpose(result, permute)

        return_val = arg.__class__(result)
        return_val.permute_qubits(application_permutation, inverse=True)
        if type(return_val) is not Operator:
            return_val.renormalize_()

        return return_val


# Factory functions for building operators

def Identity(d=1):
    """
    Produce the `d`-qubit identity operator :math:`I^{\\otimes d}`.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, 1.0 + 0.0j]])).tensor_power(d)


def PauliX(d=1):
    """
    Produce the `d`-qubit Pauli X operator :math:`X^{\\otimes d}`,
    or `not` gate.
    Maps: \|0⟩ -> \|1⟩, \|1⟩ -> \|0⟩.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator(np.array([[0.0j, 1.0 + 0.0j],
                              [1.0 + 0.0j, 0.0j]])).tensor_power(d)


def PauliY(d=1):
    """
    Produce the `d`-qubit Pauli Y operator :math:`Y^{\\otimes d}`.
    Maps: \|0⟩ -> `i` \|1⟩, \|1⟩ -> -`i` \|0⟩.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator(np.array([[0.0j, -1.0j],
                              [1.0j, 0.0j]])).tensor_power(d)


def PauliZ(d=1):
    """
    Produce the `d`-qubit Pauli Z operator :math:`Z^{\\otimes d}`,
    or phase inverter.
    Maps: \|0⟩ -> \|0⟩, \|1⟩ -> -\|1⟩.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """
    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, -1.0 + 0.0j]])).tensor_power(d)


def Hadamard(d=1):
    """
    Produce the `d`-qubit Hadamard operator :math:`H^{\\otimes d}`.
    Maps: \|0⟩ -> :math:`\\frac{1}{\\sqrt{2}}` (\|0⟩+\|1⟩) = \|+⟩,
    \|1⟩ -> :math:`\\frac{1}{\\sqrt{2}}` (\|0⟩-\|1⟩) = \|-⟩.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """
    return Operator(1/np.sqrt(2) *
        np.array([[1.0 + 0.0j,  1.0 + 0.0j],
                  [1.0 + 0.0j, -1.0 + 0.0j]])).tensor_power(d)


def Phase(d=1):
    """
    Produce the `d`-qubit Phase operator S.
    Maps: \|0⟩ -> \|0⟩, \|1⟩ -> i\|1⟩.
    Note that :math:`S^2 = Z`.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, 1.0j]])).tensor_power(d)
def PiBy8(d=1):
    """
    Produce the `d`-qubit :math:`\pi/8` operator T.
    Maps: \|0⟩ -> \|0⟩, \|1⟩ -> :math:`e^{i\pi/4}` \|1⟩.
    Note that :math:`T^2 = S`, where S is the phase gate,
    and :math:`T^4 = Z`.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, Rotation
    RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator(np.array([[1.0 + 0.0j, 0.0j],
                              [0.0j, np.exp(1j * np.pi/4)]])).tensor_power(d)

def Rotation(v, theta):
    """
    Produce the single-qubit rotation operator.
    In terms of the Bloch sphere picture of the qubit state, the
    operator rotates a state through angle :math:`\theta` around vector v.

    Parameters
    ----------
    v : list of float
        A real 3D unit vector around which the qubit's Bloch vector is to be rotated.
    theta : float
        The angle through which the qubit's Bloch vector is to be rotated.

    Returns
    -------
    Operator
        A rank 2 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase
    PiBy8, RotationX, RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    v = np.array(v)
    if v.shape != (3,) or abs(v.dot(v) - 1.0) > 1e-8 or not np.all(np.isreal(v)):
        raise ValueError('Rotation vector v should be a 3D real unit vector.')

    return np.cos(theta/2) * Identity() - 1j * np.sin(theta/2) * (
        v[0] * PauliX() + v[1] * PauliY() + v[2] * PauliZ())


def RotationX(theta):
    """
    Produce the single-qubit X-rotation operator.
    In terms of the Bloch sphere picture of the qubit state, the
    operator rotates a state through angle theta around the x-axis.

    Parameters
    ----------
    theta : float
        The angle through which the qubit's Bloch vector is to be rotated.

    Returns
    -------
    Operator
        A rank 2 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationY, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Rotation([1., 0., 0.], theta)


def RotationY(theta):
    """
    Produce the single-qubit Y-rotation operator.
    In terms of the Bloch sphere picture of the qubit state, the
    operator rotates a state through angle theta around the y-axis.

    Parameters
    ----------
    theta : float
        The angle through which the qubit's Bloch vector is to be rotated.

    Returns
    -------
    Operator
        A rank 2 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationZ, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Rotation([0., 1., 0.], theta)


def RotationZ(theta):
    """
    Produce the single-qubit Z-rotation operator.
    In terms of the Bloch sphere picture of the qubit state, the
    operator rotates a state through angle theta around the z-axis.

    Parameters
    ----------
    theta : float
        The angle through which the qubit's Bloch vector is to be rotated.

    Returns
    -------
    Operator
        A rank 2 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, SqrtNot, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Rotation([0., 0., 1.], theta)


def SqrtNot(d=1):
    """
    Produce the `d`-qubit operator that is the square root of the
    `d`-qubit NOT or :py:func:`PauliX` operator, i.e.,
    :math:`\\sqrt{\\texttt{NOT}}(\\sqrt{\\texttt{NOT}}) = X`.

    Parameters
    ----------
    d : int
        The number of qubits described by the state vector on which
        the produced operator will act.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, CNOT, Toffoli
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator(0.5 * np.array([[1 + 1j, 1 - 1j],
                                    [1 - 1j, 1 + 1j]])).tensor_power(d)


def CNOT():
    """
    Produce the two-qubit CNOT operator, which flips the second bit
    if the first bit is set.
    Maps \|00⟩ -> \|00⟩, \|01⟩ -> \|01⟩, \|10⟩ -> \|01⟩, \|11⟩ -> \|10⟩.

    Returns
    -------
    Operator
        A rank 4 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, Toffoli, SqrtNot
    Swap, SqrtSwap, ControlledU, U_f
    """

    return Operator((1.0 + 0.0j) *  np.array([[[[ 1.0, 0.0],
                                                [ 0.0, 1.0]],
                                               [[ 0.0, 0.0],
                                                [ 0.0, 0.0]]],
                                              [[[ 0.0, 0.0],
                                                [ 0.0, 0.0]],
                                               [[ 0.0, 1.0],
                                                [ 1.0, 0.0]]]]))


def Toffoli():
    """
    Produce the three-qubit Toffoli operator, which flips the third bit
    if the first two bits are set.
    Maps \|110⟩ -> \|111⟩, \|111⟩ -> \|110⟩, and otherwise acts as the
    identity.

    Returns
    -------
    Operator
        A rank 6 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, CNOT, SqrtNot
    Swap, SqrtSwap, ControlledU, U_f
    """

    d = 3
    shape = [2] * 2 * d
    t = np.zeros(shape, dtype=np.complex128)

    # Fill in the operator as the Identity operator.
    t[:] = Identity(d)[:]
    # In the case that the first two bits are set, it acts on the third
    # bit as the PauliX operator.
    t[:, 1, :, 1, ...] = (Identity(2) * PauliX())[:, 1, :, 1]

    return Operator(t)


def Swap():
    """
    Produce the two-qubit SWAP operator, which swaps two bits.
    Maps \|00⟩ -> \|00⟩, \|01⟩ -> \|10⟩, \|10⟩ -> \|01⟩, \|11⟩ -> \|11⟩.

    Returns
    -------
    Operator
        A rank 4 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot
    CNOT, Toffoli, SqrtSwap, ControlledU, U_f
    """

    return Operator((1.0 + 0.0j) *  np.array([[[[ 1.0, 0.0],
                                                [ 0.0, 0.0]],
                                               [[ 0.0, 0.0],
                                                [ 1.0, 0.0]]],
                                              [[[ 0.0, 1.0],
                                                [ 0.0, 0.0]],
                                               [[ 0.0, 0.0],
                                                [ 0.0, 1.0]]]]))


def SqrtSwap():
    """
    Produce the two-qubit operator that is the square root of the
    :py:func:`.Swap` operator, i.e.,
    :math:`\\sqrt{\\texttt{SWAP}}(\\sqrt{\\texttt{SWAP}}) = \\texttt{SWAP}`.

    Returns
    -------
    Operator
        A rank 4 tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot
    CNOT, Toffoli, Swap, ControlledU, U_f
    """

    return Operator(np.array([[[[ 1.0,                 0.0],
                                [ 0.0,      0.5 * (1 + 1j)]],
                               [[ 0.0,                 0.0],
                                [ 0.5 * (1 - 1j),      0.0]]],
                              [[[ 0.0,       0.5 * (1 - 1j)],
                                [ 0.0,                 0.0]],
                               [[ 0.5 * (1 + 1j),      0.0],
                                [ 0.0,                 1.0]]]]))


def ControlledU(U):
    """
    Produce a Controlled-U operator, an operator for a `d` + 1 qubit
    system where the supplied U is an operator for a `d` qubit system.
    If the first bit is set, apply U to the state for the remaining
    bits.

    Parameters
    ----------
    U : Operator
        The operator to be conditionally applied.

    Returns
    -------
    Operator
        A tensor whose rank is the rank of U plus 2,
        describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot
    CNOT, Toffoli, Swap, SqrtSwap, U_f
    """

    d = U.rank // 2 + 1
    shape = [2] * 2 * d
    t = np.zeros(shape, dtype=np.complex128)

    # If the first bit is zero, fill in as the identity operator.
    t[:, 0, ...] = Identity(d)[:, 0, ...]
    # Else, fill in as Identity tensored with U (Identity for the first bit,
    # which remains unchanged.
    t[:, 1, ...] = (Identity() * U)[:, 1, ...]
    return Operator(t)


def U_f(f, d):
    """
    Produce a U_f operator, an operator for a `d` qubit
    system that flips the last bit based on the outcome of a supplied
    boolean function :math:`f: [0, 1]^{d-1} \\to [0, 1]` applied to the
    first `d` - 1 bits.

    Parameters
    ----------
    f : function
        The boolean function used to conditionally flip the last bit.

    Returns
    -------
    Operator
        A rank `2d` tensor describing the operator.

    See Also
    --------
    Identity, PauliX, PauliY, PauliZ, Hadamard, Phase, PiBy8, Rotation
    RotationX, RotationY, RotationZ, SqrtNot
    CNOT, Toffoli, Swap, SqrtSwap, ControlledU
    """
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
