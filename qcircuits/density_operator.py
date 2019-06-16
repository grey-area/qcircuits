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

    # TODO call this from the measure method
    def _measurement_probabilites(self, qubit_indices):
        # Put non-measured qubits up front
        unmeasured_indices = list(set(range(self.rank // 2)) - set(qubit_indices))
        permute = unmeasured_indices + qubit_indices

        t = self._permuted_tensor(permute)

        # TODO do this in a single operation
        for _ in range(len(unmeasured_indices)):
            t = np.sum(t[[0, 1], [0, 1], ...], axis=0)

        ps = np.diag(OperatorBase._tensor_to_matrix(t))
        return ps

    def measure(self, qubit_indices=None):
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
            qubit_indices = range(self.rank // 2)

        qubit_indices = list(qubit_indices)

        if min(qubit_indices) < 0 or max(qubit_indices) >= self.rank // 2:
            raise ValueError('Trying to measure qubit index i not 0<=i<d, '
                             'where d is the rank of the state vector.')

        if len(qubit_indices) != len(set(qubit_indices)):
            raise ValueError('Qubit indices list contains repeated elements.')

        # Put non-measured qubits up front
        unmeasured_indices = list(set(range(self.rank // 2)) - set(qubit_indices))
        permute = unmeasured_indices + qubit_indices

        t = self._permuted_tensor(permute)

        # TODO do this in a single operation
        for _ in range(len(unmeasured_indices)):
            t = np.sum(t[[0, 1], [0, 1], ...], axis=0)

        ps = np.real(np.diag(OperatorBase._tensor_to_matrix(t)))

        num_outcomes = 2**len(qubit_indices)
        outcome = np.random.choice(num_outcomes, p=ps)
        bits = tuple([outcome >> i & 1 for i in range(len(qubit_indices)-1, -1, -1)])

        # TODO post-measurement state

        if int_arg:
            bits = bits[0]

        # TODO renormalize

        return bits