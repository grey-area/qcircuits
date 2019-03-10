"""
The state module contains the State class, instances of which represent quantum states of multi-qubit systems, and factory functions for creating specific quantum states.

Each of the factory functions (but not the State class) is aliased at the top-level module, so that, for example, one can call ``qcircuits.zeros()``
instead of ``qcircuits.state.zeros()``.
"""


import numpy as np

from qcircuits.tensors import Tensor
import qcircuits.operators as operators


class State(Tensor):
    """
    A container class for a tensor representing the state of a
    quantum system, and associated methods.

    Parameters
    ----------
    tensor : numpy complex128 multidimensional array
        The tensor representing the quantum state, giving the
        probability amplitudes.
    """

    def __init__(self, tensor):
        super().__init__(tensor)

        if abs(np.sum(self.probabilities) - 1.0) > 1e-4:
            raise RuntimeError('Vector is not a unit vector.')

    def __repr__(self):
        return 'State vector for {}-rank state space.'.format(self.rank)

    def __str__(self):
        s = self.__repr__() + ' Tensor:\n'
        s += super().__str__()
        return s

    def dot(self, arg):
        """
        Give the dot product between this and another State.

        Parameters
        ----------
        arg : State
            The state with which we take the dot product.

        Returns
        -------
        complex
            The dot product with State arg.
        """

        return np.sum(np.conj(self._t) * arg._t)

    @property
    def probabilities(self):
        """
        Get the probability of observing each computational basis
        vector upon making a measurement.

        Returns
        -------
        numpy float64 multidimensional array
            The probability associated with each computational basis
            vector.
        """

        probs = np.real(np.conj(self._t) * self._t)
        assert abs(np.sum(probs) - 1.0) < 1e-4, ('State probabilities'
                                                 'do not sum to 1.')
        return probs

    @property
    def amplitudes(self):
        """
        Get the state tensor, i.e., the probability amplitudes.

        Returns
        -------
        numpy complex128 multidimensional array
            The probability amplitudes of the state.
        """

        amp = np.copy(self._t)
        amp.flags.writeable = False
        return amp

    # TODO measure wrt more than 1 qubit?
    # TODO option to reduce state vector or not
    # TODO modify the state, rather than returning a state
    def measure(self, qubit_index):
        """
        Measure the state with respect to the computational bases
        of the qubit indicated by `qubit_index`.

        Parameters
        ----------
        qubit_index : int
            Indicates the qubit whose computational bases the
            measurement of the state will be made for.

        Returns
        -------
        int
            The measurement outcome for the qubit.
        State
            The state vector after the measurement, with the measured
            qubit removed from the state (i.e., with the rank of the
            underlying tensor reduced by 1).
        """

        if qubit_index < 0 or qubit_index >= self.rank:
            raise ValueError('Trying to measure qubit index i not 0<=i<d, '
                             'where d is the rank of the state vector.')

        state = np.swapaxes(self._t, 0, qubit_index)
        ps = np.reshape(np.real(state * np.conj(state)), (2, -1)).sum(axis=1)
        bit = np.random.choice([0, 1], p=ps)
        p = ps[bit]
        state = state[bit, ...] / np.sqrt(p)
        return bit, State(state)


# Factory functions for building States

def qubit(*, alpha=None, beta=None,
          theta=None, phi=None,
          global_phase=0.0):
    """
    Produce a given state for a single qubit.

    Parameters
    ----------
    alpha : float
        The probability amplitude for the \|0⟩ basis vector.
        Should not be specified in conjunction with theta or phi.
    beta : float
        The probability amplitude for the \|1⟩ basis vector.
        Should not be specified in conjunction with theta or phi.
    theta : float
        The angle of the state between \|0⟩ and \|1⟩ on the Bloch sphere.
        Should not be specified in conjunction with alpha or beta.
    phi : float
        The phase of the \|1⟩ component.
        Should not be specified in conjunction with alpha or beta.
    global_phase : float
        A global phase applied to the state.

    Returns
    -------
    State
        A rank 1 tensor describing the state of a single qubit.

    See Also
    --------
    zeros, ones, bitstring
    positive_superposition
    bell_state
    """

    # Either alpha and beta should be specified, or theta and phi,
    # but not both
    if all([v is not None for v in [alpha, beta]]) and all([v is None for v in [theta, phi]]):
        # alpha and beta specify the amplitudes of |0⟩ and |1⟩ directly
        tensor = np.zeros(2, dtype=np.complex128)
        tensor[0] = alpha
        tensor[1] = beta
    elif all([v is not None for v in [theta, phi]]) and all([v is None for v in [alpha, beta]]):
        # theta specifies the angle from |0⟩ to |1⟩ on the Bloch sphere
        # phi specifies the phase on |1⟩
        tensor = np.zeros(2, dtype=np.complex128)
        alpha = np.cos(theta/2)
        beta = np.sin(theta/2) * np.exp(1j * phi)
        tensor[0] = alpha
        tensor[1] = beta
    else:
        raise ValueError('Incorrect combination of arguments for qubit. '
                         'Supply alpha and beta, or theta and phi.')

    return State(tensor * np.exp(1j * global_phase))


def zeros(d=1):
    """
    Produce the all-zero computational basis vector for `d` qubits.
    I.e., produces :math:`|0⟩^{\otimes d}`.

    Parameters
    ----------
    d : int
        The number of qubits `d` for which we produce a computational
        basis vector, and the rank of the produced tensor.

    Returns
    -------
    State
        A `d`-rank tensor describing the all-zero `d`-qubit
        computational basis vector, :math:`|0⟩^{\otimes d}`

    See Also
    --------
    qubit, ones, bitstring
    positive_superposition
    bell_state
    """

    if d < 1:
        raise ValueError('Rank must be at least 1.')

    shape = [2] * d
    t = np.zeros(shape, dtype=np.complex128)
    t.flat[0] = 1
    return State(t)


def ones(d=1):
    """
    Produce the all-one computational basis vector for `d` qubits.
    I.e., produces :math:`|1⟩^{\otimes d}`.

    Parameters
    ----------
    d : int
        The number of qubits `d` for which we produce a computational
        basis vector, and the rank of the produced tensor.

    Returns
    -------
    State
        A `d`-rank tensor describing the all-one `d`-qubit
        computational basis vector, :math:`|1⟩^{\otimes d}`

    See Also
    --------
    qubit, zeros, bitstring
    positive_superposition
    bell_state
    """

    if d < 1:
        raise ValueError('Rank must be at least 1.')

    shape = [2] * d
    t = np.zeros(shape, dtype=np.complex128)
    t.flat[-1] = 1
    return State(t)


def bitstring(*bits):
    """
    Produce a computational basis state from a given bit sequence.

    Parameters
    ----------
    bits
        A variable number of arguments each in {0, 1}.

    Returns
    -------
    State
        If `d` arguments are supplied, returns a rank `d` tensor
        describing the computational basis state given by the sequence
        of input bits.

    See Also
    --------
    qubit, zeros, ones,
    positive_superposition
    bell_state
    """

    d = len(bits)
    if d == 0:
        raise ValueError('Rank must be at least 1.')

    shape = [2] * d
    t = np.zeros(shape, dtype=np.complex128)
    t[bits] = 1
    return State(t)


def positive_superposition(d=1):
    """
    Produce the positive superposition for a `d` qubit system, i.e.,
    the state resulting from applying the
    :py:func:`.Hadamard` gate to each of `d` \|0⟩
    computational basis states,
    :math:`H^{\otimes d}(|0⟩^{\otimes d}) = |+⟩^{\otimes d}.`

    Parameters
    ----------
    d : int
        The number of qubits `d` for the state.

    Returns
    -------
    State
        A `d`-rank tensor describing the positive superposition for
        a `d`-qubit system, :math:`|+⟩^{\otimes d}.`

    See Also
    --------
    qubit, zeros, ones, bitstring
    bell_state
    """
    if d < 1:
        raise ValueError('Rank must be at least 1.')

    H = operators.Hadamard(d)
    x = zeros(d)
    return H(x)


def bell_state(a=0, b=0):
    """
    Produce one of the four Bell states for a two-qubit system,
    :math:`|\\beta_{ab}⟩`.

    Parameters
    ----------
    a : {0, 1}
        The computational basis state of the first qubit before
        entanglement.
    b : {0, 1}
        The computational basis state of the second qubit before
        entanglement.

    Returns
    -------
    State
        A rank 2 tensor describing one of the four two-qubit Bell
        states. E.g.,
        :math:`|\\beta_{00}⟩ = \\frac{1}{\\sqrt{2}} (|00⟩ + |11⟩)`

    See Also
    --------
    qubit, zeros, ones, bitstring
    positive_superposition
    """
    if a not in [0, 1] or b not in [0, 1]:
        raise ValueError('Bell state arguments are bits, and must be 0 or 1.')

    phi = bitstring(a, b)
    phi = operators.Hadamard()(phi, qubit_indices=[0])

    return operators.CNOT()(phi)
