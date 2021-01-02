"""
The density_operator module contains the DensityOperator class,
instances of which represent mixed states of multi-qubit systems,
and functionality for creating density operators from mixtures of
pure :py:class:`.State` objects.

The DensityOperator class is
aliased at the top-level module, so that one can call
``qcircuits.DensityOperator()`` instead of
``qcircuits.state.DensityOperator()``.
"""


import numpy as np

from qcircuits.operators import OperatorBase


class DensityOperator(OperatorBase):
    """
    A density operator representing a mixed state for
    a quantum system, and associated methods.

    Parameters
    ----------
    tensor : numpy complex128 multidimensional array
        The tensor representing the operator.
    """

    def __init__(self, tensor):
        super().__init__(tensor)
        # TODO check positive, trace 1 (maybe only check when applying/measuring?)

    def __repr__(self):
        s = 'DensityOperator('
        s += super().__str__().replace('\n', '\n' + ' ' * len(s))
        s += ')'
        return s

    def __str__(self):
        s = 'Density operator for {}-qubit state space.'.format(self.rank // 2)
        s += ' Tensor:\n'
        s += super().__str__()
        return s

    @staticmethod
    def from_ensemble(states, ps=None):
        """
        Produce a density operator from an ensemble, i.e.,
        from a list of states and an equal-length list of probabilities
        that sum to one, representing classical uncertainty of the state.
        If the probabilities are not supplied, a uniform distribution
        is assumed.

        Parameters
        ----------
        states: list of State
            A list of states.
        ps: list of float
            A list of probabilities summing to one.

        Returns
        -------
        DensityOperator
            A d-qubit density operator representing a mixed state.

        """

        d = len(states)

        if ps is None:
            ps = np.ones(d) / d

        if len(ps) != len(states):
            raise ValueError('Probabilities list ps should be ' \
                             'same length as states list.')

        if len(states) == 0:
            raise ValueError('Provide at least one state.')

        if not np.isclose(np.sum(ps), 1):
            raise ValueError('Probabilities should sum to 1.')

        shape = None
        t = None

        for p, s in zip(ps, states):
            outer_product = DensityOperator._tensor_from_state_outer_product(s)

            if t is None:
                t = p * outer_product
                shape = t.shape
            else:
                if outer_product.shape != shape:
                    raise ValueError('Pure state dimensionalities do not match.')
                t += p * outer_product

        return DensityOperator(t)

    @staticmethod
    def _tensor_from_state_outer_product(state):
        result = np.tensordot(np.conj(state._t), state._t, axes=0)
        d = len(result.shape)
        permutation = [v for pair in zip(range(d//2, d), range(0, d//2)) for v in pair]

        return result.transpose(permutation)

    def _measurement_probabilities(self, qubit_indices):
        # Put non-measured qubits up front
        unmeasured_indices = list(
            set(range(self.rank // 2)) - set(qubit_indices)
        )
        t = self._permuted_tensor(unmeasured_indices + qubit_indices)

        # Trace out unmeasured bits
        # TODO do this in a single operation
        for _ in range(len(unmeasured_indices)):
            t = np.sum(t[[0, 1], [0, 1], ...], axis=0)

        ps = np.real(np.diag(OperatorBase._tensor_to_matrix(t)))
        return ps, unmeasured_indices

    def purify(self):
        """
        If this is a density operator for a :math:`d` qubit system, produce
        a pure state for a :math:`2d` qubit system with the same measurement
        probabilities for the first :math:`d` qubits, and whose reduced density
        operator for the first :math:`d` qubits is equal to the original
        density operator.

        Returns
        -------
        State
            A `2d` qubit pure state.
        """

        from qcircuits.state import State

        eigvals, eigvecs = np.linalg.eig(self.to_matrix())
        pure_column = (np.sqrt(np.expand_dims(eigvals, 0)) * eigvecs).flatten()
        return State.from_column_vector(pure_column)

    def measure(self, qubit_indices=None, remove=False):
        """
        Measure the state with respect to the computational bases
        of the qubits indicated by `qubit_indices`.
        Measuring a state will modify the state in-place.
        If no indices are indicated, the whole state is measured.

        Parameters
        ----------
        qubit_indices : int or iterable
            An index or indices indicating the qubit(s) whose
            computational bases the measurement of the state will be
            made with respect to. If no `qubit_indices` are given,
            the whole state is measured.
        remove : bool
            Indicates whether the measured qubits should be removed from
            the density operator.

        Returns
        -------
        int or tuple of int
            The measurement outcomes for the measured qubit(s).
            If the `qubit_indices` parameter is supplied as an int,
            an int is returned, otherwise a tuple.
        """

        from qcircuits.state import bitstring

        # TODO check positive with trace 1.

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
            qubit_indices = range(self.rank // 2)

        qubit_indices = list(qubit_indices)

        if qubit_indices == []:
            raise ValueError('Must measure at least one qubit.')

        if min(qubit_indices) < 0 or max(qubit_indices) >= self.rank // 2:
            raise ValueError('Trying to measure qubit index i not 0<=i<d, '
                             'where d is the rank of the state vector.')

        if len(qubit_indices) != len(set(qubit_indices)):
            raise ValueError('Qubit indices list contains repeated elements.')

        # Get measurement outcome probabilities
        ps, unmeasured_indices = self._measurement_probabilities(qubit_indices)

        num_outcomes = 2**len(qubit_indices)
        outcome = np.random.choice(num_outcomes, p=ps)
        bits = tuple(
            [outcome >> i & 1 for i in range(len(qubit_indices)-1, -1, -1)]
        )

        # Post-measurement state
        permute = qubit_indices + unmeasured_indices
        t = self._permuted_tensor(permute)
        idx = tuple(np.repeat(bits, 2))
        # Index in and renormalize
        self._t = t[idx] / ps[outcome]

        # If the measured qubits are not to be removed from the state
        # then insert them back in
        if not remove:
            measured_t = DensityOperator._tensor_from_state_outer_product(
                bitstring(*bits)
            )
            self._t = np.tensordot(measured_t, self._t, axes=0)
            self.permute_qubits(permute, inverse=True)

        if int_arg:
            bits = bits[0]

        # TODO renormalize

        return bits