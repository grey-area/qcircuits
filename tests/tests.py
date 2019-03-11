import unittest

import numpy as np
from scipy.stats import unitary_group, dirichlet

import qcircuits as qc


def max_absolute_difference(T1, T2):
    return np.max(np.abs(T1[:] - T2[:]))


def random_unitary_operator(d):
    num_basis_vectors = 2**d
    M = unitary_group.rvs(num_basis_vectors)
    permute = [0] * 2 * d
    permute[::2] = range(d)
    permute[1::2] = range(d, 2*d)

    return qc.operators.Operator(M.reshape([2] * 2 * d).transpose(permute))

def random_state(d):
    num_basis_vectors = 2**d
    shape = [2] * d
    real_part = np.sqrt(dirichlet(alpha=[1]*num_basis_vectors).rvs()[0, :])
    imag_part = np.exp(1j * np.random.uniform(0, 2*np.pi, size=num_basis_vectors))
    amplitudes = (real_part * imag_part).reshape(shape)
    return qc.state.State(amplitudes)


class OperatorCompositionTest(unittest.TestCase):
    def test_operator_composition(self):
        for test_i in range(10):
            d = np.random.randint(3, 7)
            U1 = random_unitary_operator(d)
            U2 = random_unitary_operator(d)
            U3 = random_unitary_operator(d)
            x = random_state(d)

            result1 = U1(U2(U3(x)))
            result2 = (U1(U2))(U3(x))
            result3 = (U1(U2(U3)))(x)
            max_diff = max_absolute_difference(result1, result2)
            self.assertTrue(max_diff < 1e-8)
            max_diff = max_absolute_difference(result1, result3)
            self.assertTrue(max_diff < 1e-8)


if __name__ == '__main__':
    unittest.main()
