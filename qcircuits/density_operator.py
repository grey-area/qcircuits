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
        permutation = [v for pair in zip(range(0, d//2), range(d//2, d)) for v in pair]

        return result.transpose(permutation)
