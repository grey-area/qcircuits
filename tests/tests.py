import unittest

import numpy as np
from scipy.stats import unitary_group, dirichlet

import qcircuits as qc


epsilon = 1e-10


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


class OperatorCompositionTests(unittest.TestCase):

    def setUp(self):
        self.num_tests = 10
        self.ds = []
        self.U1s = []

        for test_i in range(self.num_tests):
            d = np.random.randint(3, 7)
            U1 = random_unitary_operator(d)
            self.ds.append(d)
            self.U1s.append(U1)

    def test_operator_composition_order(self):
        """
        For two operators U1 and U2, and state x, we should have
        (U1(U2))(x) = U1(U2(x))
        """

        for test_i, (d, U1) in enumerate(zip(self.ds, self.U1s)):
            U2 = random_unitary_operator(d)
            U3 = random_unitary_operator(d)
            x = random_state(d)

            result1 = U1(U2(U3(x)))
            result2 = (U1(U2))(U3(x))
            result3 = (U1(U2(U3)))(x)
            max_diff = max_absolute_difference(result1, result2)
            self.assertTrue(max_diff < epsilon)
            max_diff = max_absolute_difference(result1, result3)
            self.assertTrue(max_diff < epsilon)

    def test_operator_identity_composition(self):
        """
        The composition of an operator with the identity operator
        should be the same as the original operator.
        """

        for test_i, (d, U) in enumerate(zip(self.ds, self.U1s)):
            I = qc.Identity(d)
            R = I(U(I))

            max_diff = max_absolute_difference(U, R)
            self.assertTrue(max_diff < epsilon)


class ApplyingToSubsetsOfQubitsTests(unittest.TestCase):

    def test_application_to_qubit_subset(self):
        """
        Instead of applying the tensor product of the Identity
        operator with another operator to a state, we can apply
        the operator to a subset of axes of the state. We should
        get the same result. We can also permute the order of
        operators in the tensor product, and correspondingly
        permute the application order.
        """

        num_tests = 10
        I = qc.Identity()

        for test_i in range(num_tests):
            d = np.random.randint(3, 7)
            num_apply_to = np.random.randint(2, d)
            apply_to_indices = np.random.choice(d, size=num_apply_to, replace=False)

            M_all = None
            Ops = []
            for qubit_i in range(d):
                if qubit_i in apply_to_indices:
                    Op = random_unitary_operator(d=1)
                else:
                    Op = I
                Ops.append(Op)
                if M_all is None:
                    M_all = Op
                else:
                    M_all = M_all * Op

            M_subset = None
            for apply_to_index in apply_to_indices:
                Op = Ops[apply_to_index]
                if M_subset is None:
                    M_subset = Op
                else:
                    M_subset = M_subset * Op

            x = random_state(d)

            result1 = M_all(x)
            result2 = M_subset(x, qubit_indices=apply_to_indices)
            max_diff = max_absolute_difference(result1, result2)
            self.assertTrue(max_diff < epsilon)


if __name__ == '__main__':
    unittest.main()
