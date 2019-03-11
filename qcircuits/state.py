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


    def measure(self, qubit_indices=None, remove=False):
        """
        Measure the state with respect to the computational bases
        of the qubits indicated by `qubit_indices`.
        Unlike operator application, with returns the resulting state,
        measuring a state will modify the state itself. If no indices
        are indicated, the whole state is measured.

        Parameters
        ----------
        qubit_indices : int or iterable
            An index or indices indicating the qubit(s) whose
            computational bases the measurement of the state will be
            made with respect to. If no `qubit_indices` are given,
            the whole state is measured.
        remove : bool
            Indicates whether the measured qubits should be removed from
            the state vector.

        Returns
        -------
        int or tuple of int
            The measurement outcomes for the measured qubit(s).
            If the `qubit_indices` parameter is supplied as an int,
            an int is returned, otherwise a tuple.
        """

        # If an int argument for qubit_indices is supplied, the return
        # value should be an int giving the single measurement outcome.
        # Otherwise, qubit_indices should be an iterable type and the
        # return type will be a tuple of measurements.
        int_arg = False
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]
            int_arg = True
        # If no indices are supplied, the whole state should be measured
        if qubit_indices is None:
            qubit_indices = range(self.rank)

        qubit_indices = list(qubit_indices)

        if min(qubit_indices) < 0 or max(qubit_indices) >= self.rank:
            raise ValueError('Trying to measure qubit index i not 0<=i<d, '
                             'where d is the rank of the state vector.')

        if len(qubit_indices) != len(set(qubit_indices)):
            raise ValueError('Qubit indices list contains repeated elements.')

        # The probability of each outcome for the qubits being measured
        num_outcomes = 2**len(qubit_indices)
        unmeasured_indices = list(set(range(self.rank)) - set(qubit_indices))
        permute = qubit_indices + unmeasured_indices
        amplitudes = np.transpose(self._t, permute)
        ps = np.reshape(np.real(amplitudes * np.conj(amplitudes)), (num_outcomes, -1)).sum(axis=1)

        # The binary representation of the measured state
        outcome = np.random.choice(num_outcomes, p=ps)
        bits = tuple([outcome >> i & 1 for i in range(len(qubit_indices)-1, -1, -1)])

        # The state of the remaining qubits post-measurement
        p = ps[outcome]
        collapsed_amplitudes = amplitudes[bits] / np.sqrt(p)

        # If the measured qubits are still to be part of the state
        # vector, put those axes back
        if remove:
            self._t = collapsed_amplitudes
        else:
            amplitudes = np.zeros_like(amplitudes)
            amplitudes[bits] = collapsed_amplitudes
            self._t = np.transpose(amplitudes, np.argsort(permute))

        # If the qubit_indices argument was an int, return the
        # single measurement as an int rather than a tuple
        if int_arg:
            bits = bits[0]

        return bits


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
