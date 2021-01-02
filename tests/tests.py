import unittest

import copy
import numpy as np
from numpy.testing import assert_allclose
from itertools import product
from scipy.stats import unitary_group, dirichlet, binom

import sys
sys.path.append('..')
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


def get_adjoint(Op):
    d = Op.rank // 2
    permute = [0] * 2 * d
    permute[::2] = range(d)
    permute[1::2] = range(d, 2*d)
    matrix_side = 2**d
    M = Op._t.transpose(np.argsort(permute)).reshape(matrix_side, matrix_side)
    M_adj = np.conj(M.T)
    op_shape = [2] * 2 * d
    M_adj = M_adj.reshape(op_shape).transpose(permute)
    return qc.operators.Operator(M_adj)


def random_state(d):
    num_basis_vectors = 2**d
    shape = [2] * d
    real_part = np.sqrt(dirichlet(alpha=[1]*num_basis_vectors).rvs()[0, :])
    imag_part = np.exp(1j * np.random.uniform(0, 2*np.pi, size=num_basis_vectors))
    amplitudes = (real_part * imag_part).reshape(shape)
    return qc.state.State(amplitudes)


def random_density_operator(d, M):
    states = [random_unitary_operator(d)(qc.zeros(d)) for _ in range(M)]
    ps = dirichlet(np.ones(M)).rvs()[0]
    return qc.DensityOperator.from_ensemble(states, ps)


def random_boolean_function(d):
    ans = np.random.choice([0, 1], 2**d)

    def f(*bits):
        index = sum(v * 2**i for i, v in enumerate(bits))

        return ans[index]

    return f


class StateUnitLengthTests(unittest.TestCase):
    """
    Test that random or defined states are of unit length.
    """

    def setUp(self):
        self.d_states = [qc.zeros, qc.ones, qc.positive_superposition]

    def test_defined_d_dim_states_unit_length(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            for state_type in self.d_states:
                state = state_type(d=d)
                diff = abs(state.probabilities.sum() - 1)
                self.assertLess(diff, epsilon)

    def test_bell_state_unit_length(self):
        for x, y in product([0, 1], repeat=2):
            state = qc.bell_state(x, y)
            diff = abs(state.probabilities.sum() - 1)
            self.assertLess(diff, epsilon)

    def test_bitstring_state_unit_length(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            bits = np.random.choice([0, 1], size=d)
            state = qc.bitstring(*bits)
            diff = abs(state.probabilities.sum() - 1)
            self.assertLess(diff, epsilon)

    def test_qubit_state_unit_length(self):
        num_tests = 10
        for test_i in range(num_tests):
            theta = np.random.normal(scale=10)
            phi = np.random.normal(scale=10)
            global_phase = np.random.normal(scale=10)
            state = qc.qubit(theta=theta, phi=phi, global_phase=global_phase)
            diff = abs(state.probabilities.sum() - 1)
            self.assertLess(diff, epsilon)

    def test_random_state_unit_length(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            state = random_state(d)
            diff = abs(state.probabilities.sum() - 1)
            self.assertLess(diff, epsilon)

    def test_tensor_product_state_unit_length(self):
        num_tests = 10
        for test_i in range(num_tests):
            d1 = np.random.randint(1, 4)
            d2 = np.random.randint(1, 4)
            state1 = random_state(d1)
            state2 = random_state(d2)
            state = state1 * state2
            diff = abs(state.probabilities.sum() - 1)
            self.assertLess(diff, epsilon)


class StateSwapPermuteTests(unittest.TestCase):
    def test_permute_reverse(self):
        """
        Test that permuting and then reversing the permutation
        results in the original state.
        """

        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            state = random_state(d)
            indices = np.arange(d)
            np.random.shuffle(indices)
            state_copy = copy.deepcopy(state)
            state.permute_qubits(indices)
            state.permute_qubits(indices, inverse=True)
            diff = max_absolute_difference(state, state_copy)
            self.assertLess(diff, epsilon)

    def test_swap_reverse(self):
        """
        Test that swapping qubits and then swapping back results
        in the original state.
        """

        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            state = random_state(d)
            i1, i2 = np.random.choice(d, replace=False, size=2)
            state_copy = copy.deepcopy(state)
            state.swap_qubits(i1, i2)
            state.swap_qubits(i1, i2)
            diff = max_absolute_difference(state, state_copy)
            self.assertLess(diff, epsilon)

    def test_permutation_of_tensor_product(self):
        """
        Test that producing a state by the tensor product of two
        states is the same as taking the tensor product in reverse
        order and then permuting.
        """

        num_tests = 10
        for test_i in range(num_tests):
            d1 = np.random.randint(3, 6)
            d2 = np.random.randint(3, 6)
            s1 = random_state(d1)
            s2 = random_state(d2)
            s = s1 * s2
            s_reverse = copy.deepcopy(s2 * s1)
            idx = list(range(d1, d1+d2)) + list(range(0, d1))
            s.permute_qubits(idx)
            self.assertLess(
                max_absolute_difference(s, s_reverse),
                epsilon)

    def test_operator_sub_application_equivalence_to_perumation(self):
        num_tests = 10
        for test_i in range(num_tests):
            state_d = np.random.randint(3, 8)
            op_d = np.random.randint(2, state_d)
            state1 = random_state(state_d)
            U = random_unitary_operator(op_d)

            application_indices = list(np.random.choice(state_d, replace=False, size=op_d))
            state = copy.deepcopy(state1)
            result1 = U(state, qubit_indices=application_indices)

            non_app_indices = sorted(list(set(range(state_d)) - set(application_indices)))
            permutation = application_indices + non_app_indices
            state = copy.deepcopy(state1)
            state.permute_qubits(permutation)
            result2 = U(state, qubit_indices=range(op_d))
            result2.permute_qubits(permutation, inverse=True)

            diff = max_absolute_difference(result1, result2)
            self.assertLess(diff, epsilon)

    def test_operator_d1_sub_application_equivalence_to_swap(self):
        num_tests = 10
        for test_i in range(num_tests):
            state_d = np.random.randint(3, 8)
            state1 = random_state(state_d)
            U = random_unitary_operator(1)

            application_index = np.random.choice(state_d)
            state = copy.deepcopy(state1)
            result1 = U(state, qubit_indices=[application_index])

            state = copy.deepcopy(state1)
            state.swap_qubits(0, application_index)
            result2 = U(state, qubit_indices=[0])
            result2.swap_qubits(0, application_index)

            diff = max_absolute_difference(result1, result2)
            self.assertLess(diff, epsilon)


class OperatorSwapPermuteTests(unittest.TestCase):
    def test_permute_reverse(self):
        """
        Test that permuting and then reversing the permutation
        results in the original operator.
        """

        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            operator = random_unitary_operator(d)
            indices = np.arange(d)
            np.random.shuffle(indices)
            operator_copy = copy.deepcopy(operator)
            operator.permute_qubits(indices)
            operator.permute_qubits(indices, inverse=True)
            diff = max_absolute_difference(operator, operator_copy)
            self.assertLess(diff, epsilon)

    def test_swap_reverse(self):
        """
        Test that swapping qubits and then swapping back results
        in the original operator.
        """

        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            op = random_unitary_operator(d)
            i1, i2 = np.random.choice(d, replace=False, size=2)
            op_copy = copy.deepcopy(op)
            op.swap_qubits(i1, i2)
            op.swap_qubits(i1, i2)
            diff = max_absolute_difference(op, op_copy)
            self.assertLess(diff, epsilon)

    def test_permutation_of_tensor_product(self):
        """
        Test that producing an operator by the tensor product of two
        operators is the same as taking the tensor product in reverse
        order and then permuting.
        """

        num_tests = 10
        for test_i in range(num_tests):
            d1 = np.random.randint(3, 6)
            d2 = np.random.randint(3, 6)
            Op1 = random_unitary_operator(d1)
            Op2 = random_unitary_operator(d2)
            Op = Op1 * Op2
            Op_reverse = copy.deepcopy(Op2 * Op1)
            idx = list(range(d1, d1+d2)) + list(range(0, d1))
            Op.permute_qubits(idx)
            self.assertLess(
                max_absolute_difference(Op, Op_reverse),
                epsilon)


class TensorProductTests(unittest.TestCase):
    def test_tensor_product(self):
        """
        Test that (A * B)(x * y) = A(x) * B(y)
        """

        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 6)
            A = random_unitary_operator(d)
            B = random_unitary_operator(d)
            x = random_state(d)
            y = random_state(d)
            R1 = (A * B)(x * y)
            R2 = A(x) * B(y)
            diff = max_absolute_difference(R1, R2)
            self.assertLess(diff, epsilon)


class OperatorApplyToSubSystemTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_application_to_subsystem(self):
        num_tests = 10
        for test_i in range(num_tests):
            d_left = np.random.randint(2, 4)
            d_op = np.random.randint(2, 4)
            d_right = np.random.randint(2, 4)
            d_total = d_left + d_op + d_right
            M = random_unitary_operator(d_total)

            I1 = qc.Identity(d_left)
            I2 = qc.Identity(d_right)
            Op = random_unitary_operator(d_op)

            R1 = (I1 * Op * I2)(M)
            R2 = Op(M, qubit_indices=range(d_left, d_left + d_op))
            self.assertLess(
                max_absolute_difference(R1, R2),
                epsilon)

    def test_same_as_applying_to_substate(self):
        num_tests = 10
        for test_i in range(num_tests):
            state_d = np.random.randint(2, 8)
            op_d = np.random.randint(1, state_d + 1)
            x = random_state(state_d)
            I = qc.Identity(state_d)
            U = random_unitary_operator(op_d)
            qubit_indices = np.random.choice(state_d, size=op_d, replace=False)

            R1 = U(copy.deepcopy(x), qubit_indices=qubit_indices)
            R2 = (U(I, qubit_indices=qubit_indices))(copy.deepcopy(x))

            self.assertLess(
                max_absolute_difference(R1, R2),
                epsilon)

    def test_same_as_applying_to_substate2(self):
        num_tests = 10
        for test_i in range(num_tests):
            state_d = np.random.randint(2, 8)
            op_d = np.random.randint(1, state_d + 1)
            x = random_state(state_d)
            M = qc.Identity(state_d)
            U = random_unitary_operator(op_d)
            qubit_indices = np.random.choice(state_d, size=op_d, replace=False)

            R1 = copy.deepcopy(U)(copy.deepcopy(M)(copy.deepcopy(x)), qubit_indices=qubit_indices)
            R2 = copy.deepcopy(U)(copy.deepcopy(M), qubit_indices=qubit_indices)(copy.deepcopy(x))

            self.assertLess(
                max_absolute_difference(R1, R2),
                epsilon)

    def test_operators_unchanged(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            op_d = np.random.randint(2, d)

            arg_Op = random_unitary_operator(d)
            Op = random_unitary_operator(op_d)
            arg_Op_copy = copy.deepcopy(arg_Op)
            Op_copy = copy.deepcopy(Op)

            qubit_indices = np.random.choice(d, size=op_d, replace=False)
            R = Op(arg_Op, qubit_indices=qubit_indices)

            self.assertLess(
                max_absolute_difference(Op, Op_copy),
                epsilon)
            self.assertLess(
                max_absolute_difference(arg_Op, arg_Op_copy),
                epsilon)


class OperatorIdentitiesTest(unittest.TestCase):

    def setUp(self):
        self.squared_equals_I_list = [
            qc.Hadamard, qc.PauliX, qc.PauliY, qc.PauliZ
        ]

    def test_squared_equals_I(self):
        for Op_type in self.squared_equals_I_list:
            I = qc.Identity()
            U = Op_type()

            diff = max_absolute_difference(U(U), I)
            self.assertLess(diff, epsilon)

        I = qc.Identity(2)
        C = qc.CNOT()
        self.assertLess(
            max_absolute_difference(
                C(C), I), epsilon)


    def test_pauli_identities(self):
        X = qc.PauliX()
        Y = qc.PauliY()
        Z = qc.PauliZ()

        self.assertLess(
            max_absolute_difference(
                X(Y), 1j*Z), epsilon)
        self.assertLess(
            max_absolute_difference(
                Y(X), -1j*Z), epsilon)
        self.assertLess(
            max_absolute_difference(
                Y(Z), 1j*X), epsilon)
        self.assertLess(
            max_absolute_difference(
                Z(Y), -1j*X), epsilon)
        self.assertLess(
            max_absolute_difference(
                Z(X), 1j*Y), epsilon)
        self.assertLess(
            max_absolute_difference(
                X(Z), -1j*Y), epsilon)

    def test_hadamard_pauli_identities(self):
        X = qc.PauliX()
        Y = qc.PauliY()
        Z = qc.PauliZ()
        H = qc.Hadamard()

        self.assertLess(
            max_absolute_difference(
                H(X(H)), Z), epsilon)
        self.assertLess(
            max_absolute_difference(
                H(Z(H)), X), epsilon)
        self.assertLess(
            max_absolute_difference(
                H(Y(H)), -Y), epsilon)
        self.assertLess(
            max_absolute_difference(
                X(Y(X)), -Y), epsilon)

    def test_CNOT_swap_identity(self):
        C = qc.CNOT()
        Sw = qc.Swap()

        Sw1 = C(C(C, qubit_indices=[1, 0]))
        self.assertLess(
            max_absolute_difference(
                Sw, Sw1), epsilon)

    def test_CNOT_hadamard_identity(self):
        C = qc.CNOT()
        H = qc.Hadamard()

        C1 = (H * H)(C(H * H, qubit_indices=[1, 0]))

        self.assertLess(
            max_absolute_difference(
                C, C1), epsilon)

    def test_phase_identities(self):
        Z = qc.PauliZ()
        S = qc.Phase()
        T = qc.PiBy8()
        H = qc.Hadamard()

        self.assertLess(
            max_absolute_difference(
                T(T), S), epsilon)
        self.assertLess(
            max_absolute_difference(
                S(S), Z), epsilon)
        self.assertLess(
            max_absolute_difference(
                T(T(T(T))), Z), epsilon)
        self.assertLess(
            max_absolute_difference(
                T,
                np.exp(1j * np.pi/8) * qc.RotationZ(np.pi/4)
            ),
            epsilon
        )
        self.assertLess(
            max_absolute_difference(
                S,
                np.exp(1j * np.pi/4) * qc.RotationZ(np.pi/2)
            ),
            epsilon
        )
        self.assertLess(
            max_absolute_difference(
                H(T(H)),
                np.exp(1j * np.pi/8) * qc.RotationX(np.pi/4)
            ),
            epsilon
        )

    def test_rotations_equal_pauli(self):
        X = qc.PauliX()
        Y = qc.PauliY()
        Z = qc.PauliZ()
        Rx = qc.RotationX(np.pi)
        Ry = qc.RotationY(np.pi)
        Rz = qc.RotationZ(np.pi)

        self.assertLess(
            max_absolute_difference(
                1j*Rx, X), epsilon)
        self.assertLess(
            max_absolute_difference(
                1j*Ry, Y), epsilon)
        self.assertLess(
            max_absolute_difference(
                1j*Rz, Z), epsilon)

    def test_phase_square(self):
        S = qc.Phase()
        Z = qc.PauliZ()

        self.assertLess(
            max_absolute_difference(
                S(S), Z), epsilon)

    def test_sqrtnot_squared_equals_X(self):
        R1 = qc.SqrtNot()(qc.SqrtNot())
        R2 = qc.PauliX()
        diff = max_absolute_difference(R1, R2)
        self.assertLess(diff, epsilon)

    def test_sqrtswap_squared_equals_swap(self):
        R1 = qc.SqrtSwap()(qc.SqrtSwap())
        R2 = qc.Swap()
        diff = max_absolute_difference(R1, R2)
        self.assertLess(diff, epsilon)


class OperatorUnitaryTests(unittest.TestCase):
    """
    Various tests to test if random or defined operators are unitary.
    """

    def setUp(self):
        self.d_ops = [
            qc.Identity, qc.PauliX, qc.PauliY, qc.PauliZ,
            qc.Hadamard, qc.Phase, qc.SqrtNot
        ]
        self.d_op_names = ['I', 'X', 'Y', 'Z', 'H', 'Phase', 'SqrtNot']
        self.non_d_ops = [
            qc.CNOT, qc.Toffoli, qc.Swap, qc.SqrtSwap
        ]
        self.non_d_op_names = ['CNOT', 'Toffoli', 'Swap', 'SqrtSwap']
        self.non_d_op_dims = [2, 3, 2, 2]

    def test_random_operator_unitary(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            Op = random_unitary_operator(d)
            Op_adj = get_adjoint(Op)
            I = qc.Identity(d)
            max_diff = max_absolute_difference(Op(Op_adj), I)
            self.assertLess(max_diff, epsilon)
            max_diff = max_absolute_difference(Op_adj(Op), I)
            self.assertLess(max_diff, epsilon)

    def test_random_operator_adjoint_equal_matrix_adjoint(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            U = random_unitary_operator(d)
            U_from_matrix_adjoint = get_adjoint(U)
            U_from_adjoint_property = U.adj
            max_diff = max_absolute_difference(U_from_adjoint_property,
                                               U_from_matrix_adjoint)
            self.assertLess(max_diff, epsilon)

    def test_tensor_product_operator_unitary(self):
        num_tests = 10
        for test_i in range(num_tests):
            d1 = np.random.randint(3, 5)
            d2 = np.random.randint(3, 5)
            d = d1 + d2
            Op1 = random_unitary_operator(d1)
            Op2 = random_unitary_operator(d2)
            Op = Op1 * Op2
            Op_adj = get_adjoint(Op)
            I = qc.Identity(d)
            max_diff = max_absolute_difference(Op(Op_adj), I)
            self.assertLess(max_diff, epsilon)
            max_diff = max_absolute_difference(Op_adj(Op), I)
            self.assertLess(max_diff, epsilon)

    def test_defined_operators_unitary(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 8)
            I = qc.Identity(d)
            for Op_type, op_name in zip(self.d_ops, self.d_op_names):
                Op = Op_type(d=d)
                Op_adj = get_adjoint(Op)
                max_diff = max_absolute_difference(Op(Op_adj), I)
                self.assertLess(max_diff, epsilon)
                max_diff = max_absolute_difference(Op_adj(Op), I)
                self.assertLess(max_diff, epsilon)
        for Op_type, op_name, d in zip(self.non_d_ops, self.non_d_op_names, self.non_d_op_dims):
            Op = Op_type()
            Op_adj = get_adjoint(Op)
            I = qc.Identity(d)
            max_diff = max_absolute_difference(Op(Op_adj), I)
            self.assertLess(max_diff, epsilon)
            max_diff = max_absolute_difference(Op_adj(Op), I)
            self.assertLess(max_diff, epsilon)

    def test_U_f_unitary(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 7)
            f = random_boolean_function(d)
            Op = qc.U_f(f, d=d+1)
            Op_adj = get_adjoint(Op)
            I = qc.Identity(d+1)
            max_diff = max_absolute_difference(Op(Op_adj), I)
            self.assertLess(max_diff, epsilon)
            max_diff = max_absolute_difference(Op_adj(Op), I)
            self.assertLess(max_diff, epsilon)

    def test_ControlledU_unitary(self):
        num_tests = 10
        for test_i in range(num_tests):
            d = np.random.randint(3, 7)
            U = random_unitary_operator(d)
            Op = qc.ControlledU(U)
            Op_adj = get_adjoint(Op)
            I = qc.Identity(d+1)
            max_diff = max_absolute_difference(Op(Op_adj), I)
            self.assertLess(max_diff, epsilon)
            max_diff = max_absolute_difference(Op_adj(Op), I)
            self.assertLess(max_diff, epsilon)


class StateAdditionScalarMultiplicationTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_superposition(self):
        x = qc.positive_superposition()
        y1 = (qc.zeros() + qc.ones()) / np.sqrt(2)
        y2 = (qc.zeros() + qc.ones()) * (1 / np.sqrt(2))
        y3 = (1 / np.sqrt(2)) * (qc.ones() + qc.zeros())
        y4 = (1 / np.sqrt(2)) * qc.ones() + qc.zeros() / np.sqrt(2)
        self.assertLess(max_absolute_difference(y1, x), epsilon)
        self.assertLess(max_absolute_difference(y2, x), epsilon)
        self.assertLess(max_absolute_difference(y3, x), epsilon)
        self.assertLess(max_absolute_difference(y4, x), epsilon)

        self.assertLess(
            max_absolute_difference(
                np.sqrt(2) * x - qc.ones(),
                qc.zeros()
            ),
            epsilon
        )


class OperatorAdditionScalarMultiplicationTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_hadamard(self):
        """
        H = (X + Y)/sqrt(2)
        """
        H = qc.Hadamard()
        X = qc.PauliX()
        Z = qc.PauliZ()
        H1 = (X + Z) / np.sqrt(2)
        max_diff = max_absolute_difference(H, H1)
        self.assertLess(max_diff, epsilon)
        H2 = (X + Z) * (1 / np.sqrt(2))
        max_diff = max_absolute_difference(H, H2)
        self.assertLess(max_diff, epsilon)
        H3 = 1 / np.sqrt(2) * (Z + X)
        max_diff = max_absolute_difference(H, H3)
        self.assertLess(max_diff, epsilon)

        self.assertLess(
            max_absolute_difference(
                np.sqrt(2) * H - Z,
                X
            ),
            epsilon
        )


class OperatorCompositionTests(unittest.TestCase):

    def setUp(self):
        self.num_tests = 10
        self.ds = []
        self.U1s = []

        for test_i in range(self.num_tests):
            d = np.random.randint(3, 8)
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
            self.assertLess(max_diff, epsilon)
            max_diff = max_absolute_difference(result1, result3)
            self.assertLess(max_diff, epsilon)

    def test_operator_identity_composition(self):
        """
        The composition of an operator with the identity operator
        should be the same as the original operator.
        """

        for test_i, (d, U) in enumerate(zip(self.ds, self.U1s)):
            I = qc.Identity(d)
            R = I(U(I))

            max_diff = max_absolute_difference(U, R)
            self.assertLess(max_diff, epsilon)


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
            d = np.random.randint(3, 8)
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
            self.assertLess(max_diff, epsilon)


class KroneckerProductTests(unittest.TestCase):

    def setUp(self):
        self.num_tests = 10
        self.ds = np.random.randint(1, 8, size=self.num_tests)
        self.states = [random_state(d) for d in self.ds]
        self.operators = [random_unitary_operator(d) for d in self.ds]

    def test_bitstring_kronecker(self):
        d = 5
        for bits in product([0, 1], repeat=d):
            bitstring = qc.bitstring(*bits)
            v1 = bitstring.to_column_vector()
            v2 = np.zeros(2**d)
            index = 0
            for b_i, b in enumerate(bits):
                index += b * 2**(d - b_i - 1)
            v2[index] = 1
            self.assertLess(max_absolute_difference(v1, v2), epsilon)

    def test_state_kronecker_reversible_random_1(self):
        num_tests = 10
        for s1 in self.states:
            v = s1.to_column_vector()
            s2 = qc.State.from_column_vector(v)
            self.assertLess(max_absolute_difference(s1, s2), epsilon)

    def test_state_kronecker_reversible_random_2(self):
        num_tests = 10
        for d in self.ds:
            size = 2**d
            v1 = np.random.normal(size=size) + 1j * np.random.normal(size=size)
            s = qc.State.from_column_vector(v1)
            v2 = s.to_column_vector()
            self.assertLess(max_absolute_difference(v1, v2), epsilon)

    def test_operator_kronecker_reversible_random_1(self):
        num_tests = 10
        for o1 in self.operators:
            M = o1.to_matrix()
            o2 = qc.Operator.from_matrix(M)
            self.assertLess(max_absolute_difference(o1, o2), epsilon)

    def test_operator_kronecker_reversible_random_2(self):
        num_tests = 10
        for d in self.ds:
            d = np.random.randint(1, 8)
            size = [2**d] * 2
            M1 = np.random.normal(size=size) + 1j * np.random.normal(size=size)
            o = qc.Operator.from_matrix(M1)
            M2 = o.to_matrix()
            self.assertLess(max_absolute_difference(M1, M2), epsilon)

    def test_kronecker_operator_state_application_1(self):
        for Op, s in zip(self.operators, self.states):
            v1 = Op(s).to_column_vector()
            M = Op.to_matrix()
            x = s.to_column_vector()
            v2 = M.dot(x)
            self.assertLess(max_absolute_difference(v1, v2), epsilon)

    def test_kronecker_operator_state_application_2(self):
        for Op, s in zip(self.operators, self.states):
            s1 = Op(s)
            M = Op.to_matrix()
            x = s.to_column_vector()
            v = M.dot(x)
            s2 = qc.State.from_column_vector(v)

            self.assertLess(max_absolute_difference(s1, s2), epsilon)


class SchmidtTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_schmidt_examples(self):
        self.assertEqual(
            qc.bell_state().schmidt_number(indices=[0]),
            2
        )

        self.assertEqual(
            qc.positive_superposition(d=2).schmidt_number(indices=[0]),
            1
        )

        x = (qc.bitstring(0, 0) + qc.bitstring(0, 1) + qc.bitstring(1, 0)) / np.sqrt(3)
        self.assertEqual(
            x.schmidt_number(indices=[0]),
            2
        )

    def test_random_schmidt_examples_unentangled(self):
        num_tests = 10
        for test_i in range(num_tests):
            d1 = np.random.randint(1, 6)
            s1 = random_state(d=d1)
            d2 = np.random.randint(1, 6)
            s2 = random_state(d=d2)
            num = (s1 * s2).schmidt_number(indices=list(range(d1)))
            self.assertEqual(num, 1)

    def test_unitarily_unchanged(self):
        num_tests = 10

        for test_i in range(num_tests):

            x = qc.positive_superposition(d=2)
            U = random_unitary_operator(d=1) * random_unitary_operator(d=1)
            self.assertEqual(
                U(x).schmidt_number(indices=[0]),
                1
            )

            x = qc.bell_state()
            U = random_unitary_operator(d=1) * random_unitary_operator(d=1)
            self.assertEqual(
                U(x).schmidt_number(indices=[0]),
                2
            )



class DensityOperatorTests(unittest.TestCase):

    def test_pure_construction_comparison(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            state = random_state(d)

            rho1 = state.density_operator()
            rho2 = qc.DensityOperator.from_ensemble([state], [1.])
            rho3 = state.reduced_density_operator(qubit_indices=list(range(d)))

            assert_allclose(rho1._t, rho2._t)
            assert_allclose(rho1._t, rho3._t)

    def test_outer_product(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            state = random_state(d)
            result1 = qc.DensityOperator._tensor_from_state_outer_product(state)
            result2 = qc.Operator.from_matrix(np.dot(
                np.expand_dims(state.to_column_vector(), 1),
                np.expand_dims(np.conj(state.to_column_vector()), 0)
            ))._t

            assert_allclose(result1, result2)

    def test_operator_application(self):
        num_tests = 10

        for test_i in range(num_tests):
            num_states = np.random.randint(1, 8)
            d = np.random.randint(1, 8)
            Op = random_unitary_operator(d)
            states = [random_state(d) for i in range(num_states)]
            ps = dirichlet(np.ones(num_states)).rvs()[0]

            rho1 = qc.DensityOperator.from_ensemble(states, ps)
            result1 = Op(rho1)

            post_states = [Op(state) for state in states]
            result2 = qc.DensityOperator.from_ensemble(post_states, ps)

            assert_allclose(result1._t, result2._t)

    def test_operator_subsystem_application(self):
        num_tests = 10

        for test_i in range(num_tests):
            num_states = np.random.randint(1, 8)
            d = np.random.randint(1, 8)
            application_d = np.random.randint(1, d + 1)
            Op = random_unitary_operator(application_d)
            application_idx = np.random.choice(d, size=application_d, replace=False)
            states = [random_state(d) for i in range(num_states)]
            ps = dirichlet(np.ones(num_states)).rvs()[0]

            rho1 = qc.DensityOperator.from_ensemble(states, ps)
            result1 = Op(rho1, qubit_indices=application_idx)

            post_states = [Op(state, qubit_indices=application_idx) for state in states]
            result2 = qc.DensityOperator.from_ensemble(post_states, ps)

            assert_allclose(result1._t, result2._t)

    def test_permutation_of_density_operators(self):
        num_tests = 10

        for test_i in range(num_tests):
            num_states = np.random.randint(1, 8)
            d = np.random.randint(2, 8)
            permutation = np.arange(d)
            np.random.shuffle(permutation)
            states = [random_state(d) for i in range(num_states)]
            ps = dirichlet(np.ones(num_states)).rvs()[0]

            rho1 = qc.DensityOperator.from_ensemble(states, ps)
            rho1.permute_qubits(permutation)

            for state in states:
                state.permute_qubits(permutation)
            rho2 = qc.DensityOperator.from_ensemble(states, ps)

            assert_allclose(rho1._t, rho2._t)

    def test_qubit_swap_of_density_operators(self):
        num_tests = 10

        for test_i in range(num_tests):
            num_states = np.random.randint(1, 8)
            d = np.random.randint(2, 8)
            idx1, idx2 = np.random.choice(d, size=2, replace=False)
            states = [random_state(d) for i in range(num_states)]
            ps = dirichlet(np.ones(num_states)).rvs()[0]

            rho1 = qc.DensityOperator.from_ensemble(states, ps)
            rho1.swap_qubits(idx1, idx2)

            for state in states:
                state.swap_qubits(idx1, idx2)
            rho2 = qc.DensityOperator.from_ensemble(states, ps)

            assert_allclose(rho1._t, rho2._t)

    def test_measurement_probabilities(self):
        num_tests = 10

        for test_i in range(num_tests):
            num_states = np.random.randint(1, 8)
            d = np.random.randint(2, 8)
            states = [random_state(d) for i in range(num_states)]
            ensemble_ps = dirichlet(np.ones(num_states)).rvs()[0]
            rho = qc.DensityOperator.from_ensemble(states, ensemble_ps)

            measurement_d = np.random.randint(1, d + 1)
            measurement_qubits = list(np.random.choice(d, size=measurement_d, replace=False))

            ps1, _ = rho._measurement_probabilities(qubit_indices=measurement_qubits)
            ps2 = np.zeros_like(ps1)
            for ensemble_p, state in zip(ensemble_ps, states):
                ps, _, _, _ = state._measurement_probabilities(qubit_indices=measurement_qubits)
                ps2 += ensemble_p * ps

            assert_allclose(ps1, ps2)

    def test_measurements_dont_change(self):
        num_tests = 10

        for test_i in range(num_tests):
            num_states = np.random.randint(1, 8)
            d = np.random.randint(2, 8)
            states = [random_state(d) for i in range(num_states)]
            ensemble_ps = dirichlet(np.ones(num_states)).rvs()[0]
            rho = qc.DensityOperator.from_ensemble(states, ensemble_ps)

            measurement_d = np.random.randint(1, d+1)
            qubit_indices = np.random.choice(d, measurement_d, replace=False)

            result1 = rho.measure(qubit_indices)
            result2 = rho.measure(qubit_indices)

            self.assertEqual(result1, result2)

    def test_qubit_subset_measurement(self):
        num_tests = 10

        for test_i in range(num_tests):
            subsystems = []
            ds = []
            for i in range(3):
                num_states = np.random.randint(1, 8)
                d = np.random.randint(1, 3)
                ds.append(d)
                states = [random_state(d) for i in range(num_states)]
                ensemble_ps = dirichlet(np.ones(num_states)).rvs()[0]
                rho = qc.DensityOperator.from_ensemble(states, ensemble_ps)
                subsystems.append(rho)

            system = subsystems[0] * subsystems[1] * subsystems[2]
            qubit_indices = list(range(ds[0], ds[0]+ds[1]))
            system.measure(qubit_indices=qubit_indices, remove=True)
            reduced_system = subsystems[0] * subsystems[2]

            assert_allclose(system._t, reduced_system._t)

    def test_computational_basis_state_measurement(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(2, 8)
            bits = np.random.randint(2, size=d)
            bits_list = list(bits)
            rho = qc.DensityOperator.from_ensemble([qc.bitstring(*bits_list)])
            measurement_d = np.random.randint(1, d+1)
            qubit_indices = np.random.choice(d, size=measurement_d, replace=False)

            measured_bits = rho.measure(qubit_indices)

            assert_allclose(
                bits[qubit_indices],
                np.array(measured_bits)
            )

    def test_positive_superposition_measurement(self):
        state = qc.positive_superposition()
        rho1 = qc.DensityOperator.from_ensemble([state])
        measurement = rho1.measure()[0]

        state = qc.bitstring(measurement)
        rho2 = qc.DensityOperator.from_ensemble([state])

        assert_allclose(rho1._t, rho2._t)

    def test_bell_state_measurement(self):
        state = qc.bell_state(0, 0)
        rho1 = qc.DensityOperator.from_ensemble([state])
        idx = np.random.randint(2)
        m = rho1.measure(qubit_indices=[idx])[0]

        state = qc.bitstring(m, m)
        rho2 = qc.DensityOperator.from_ensemble([state])

        assert_allclose(rho1._t, rho2._t)

    def test_bell_state_measurement2(self):
        state = qc.bell_state(0, 0)
        rho1 = qc.DensityOperator.from_ensemble([state])
        idx = np.random.randint(2)
        m = rho1.measure(qubit_indices=[idx], remove=True)[0]

        state = qc.bitstring(m)
        rho2 = qc.DensityOperator.from_ensemble([state])

        assert_allclose(rho1._t, rho2._t)

    def test_state_density_operator_consistent(self):
        num_tests = 10

        for test_i in range(num_tests):
            for remove in [False, True]:

                d = np.random.randint(1, 8)
                state = random_state(d)
                rho = qc.DensityOperator.from_ensemble([state])

                measure_idx = np.random.randint(d)
                density_m = rho.measure(qubit_indices=measure_idx, remove=remove)

                state_m = None
                state1 = None
                while state_m != density_m:
                    state1 = copy.deepcopy(state)
                    state_m = state1.measure(measure_idx, remove=remove)

                rho2 = qc.DensityOperator.from_ensemble([state1])

                assert_allclose(rho._t, rho2._t)

    def test_state_density_operator_consistent2(self):
        num_tests = 10

        for test_i in range(num_tests):
            for remove in [False, True]:

                d = np.random.randint(2, 8)
                state = random_state(d)
                rho = qc.DensityOperator.from_ensemble([state])

                measure_idx = np.random.choice(d, size=2, replace=False)
                density_m = rho.measure(qubit_indices=measure_idx, remove=remove)

                state_m = None
                state1 = None
                while state_m != density_m:
                    state1 = copy.deepcopy(state)
                    state_m = state1.measure(measure_idx, remove=remove)

                rho2 = qc.DensityOperator.from_ensemble([state1])

                assert_allclose(rho._t, rho2._t)

    # We can use Bayes rule and knowledge of the probability of
    # a given measurement outcome for each state in the ensemble
    # to compute a posterior distribution over the states, given
    # a measurement. We can compute another density operator from
    # the posterior, and compare it to the result of doing the
    # measurement on the density operator directly
    def test_post_measurement_state_for_mixed_states(self):
        num_tests = 10

        for test_i in range(num_tests):
            for remove in [False, True]:
                num_states = np.random.randint(1, 8)
                d = np.random.randint(2, 8)
                states = [random_state(d) for i in range(num_states)]
                ensemble_ps = dirichlet(np.ones(num_states)).rvs()[0]
                rho = qc.DensityOperator.from_ensemble(states, ensemble_ps)

                max_measurement_d = min(d - 1, 4)
                measurement_d = np.random.randint(1, max_measurement_d + 1)
                measure_idx = list(np.random.choice(d, size=measurement_d, replace=False))

                density_m = rho.measure(measure_idx, remove=remove)
                outcome = int(''.join([str(i) for i in density_m]), 2)

                p_m_given_i = []
                for state in states:
                    ps, _, _, _ = state._measurement_probabilities(measure_idx)
                    p_m_given_i.append(ps[outcome])
                p_m_given_i = np.array(p_m_given_i)

                post_measurement_states = []
                for state in states:
                    state_m = None
                    state1 = None
                    while state_m != density_m:
                        state1 = copy.deepcopy(state)
                        state_m = state1.measure(measure_idx, remove=remove)
                    post_measurement_states.append(state1)

                p_i_given_m = p_m_given_i * ensemble_ps
                p_i_given_m /= np.sum(p_i_given_m)

                rho2 = qc.DensityOperator.from_ensemble(
                    post_measurement_states,
                    p_i_given_m
                )

                assert_allclose(rho._t, rho2._t)


class ReducedDensityAndPurificationTests(unittest.TestCase):

    def test_tracing_out_product_state(self):
        num_tests = 10

        for _ in range(num_tests):
            d1 = np.random.randint(1, 4)
            d2 = np.random.randint(1, 4)
            d3 = np.random.randint(1, 4)
            M1 = np.random.randint(2, 10)
            M2 = np.random.randint(2, 10)
            M3 = np.random.randint(2, 10)

            rho1 = random_density_operator(d1, M1)
            rho2 = random_density_operator(d2, M2)
            rho3 = random_density_operator(d3, M3)

            rho_combined = rho1 * rho2 * rho3
            rho_reduced1 = rho_combined.reduced_density_operator(list(range(d1)))
            rho_reduced2 = rho_combined.reduced_density_operator(list(range(d1, d1 + d2)))
            rho_reduced3 = rho_combined.reduced_density_operator(list(range(d1 + d2, d1 + d2 + d3)))

            assert_allclose(rho1._t, rho_reduced1._t)
            assert_allclose(rho2._t, rho_reduced2._t)
            assert_allclose(rho3._t, rho_reduced3._t)

    def test_tracing_out_product_state2(self):
        num_tests = 10

        for _ in range(num_tests):
            d1 = np.random.randint(1, 4)
            d2 = np.random.randint(1, 4)
            d3 = np.random.randint(1, 4)

            state1 = random_state(d1)
            state2 = random_state(d2)
            state3 = random_state(d3)

            rho1 = state1.reduced_density_operator(list(range(d1)))
            rho2 = state2.reduced_density_operator(list(range(d2)))
            rho3 = state3.reduced_density_operator(list(range(d3)))

            state_combined = state1 * state2 * state3

            rho_reduced1 = state_combined.reduced_density_operator(list(range(d1)))
            rho_reduced2 = state_combined.reduced_density_operator(list(range(d1, d1 + d2)))
            rho_reduced3 = state_combined.reduced_density_operator(list(range(d1 + d2, d1 + d2 + d3)))

            assert_allclose(rho1._t, rho_reduced1._t)
            assert_allclose(rho2._t, rho_reduced2._t)
            assert_allclose(rho3._t, rho_reduced3._t)

    def test_retain_all_qubits(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 5)
            M = np.random.randint(2, 10)
            rho = random_density_operator(d, M)
            rho2 = rho.reduced_density_operator(list(range(d)))
            assert_allclose(rho._t, rho2._t)

    def test_purify_then_trace_out(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 5)
            M = np.random.randint(2, 10)

            rho = random_density_operator(d, M)
            pure_state = rho.purify()
            rho2 = pure_state.reduced_density_operator(list(range(d)))

            assert_allclose(rho._t, rho2._t)

    def test_measurement_probability_compatability1(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            state = random_state(d)
            rho = qc.DensityOperator.from_ensemble([state], [1.])

            ps1 = state._measurement_probabilities(qubit_indices=list(range(d)))[0]
            ps2 = rho._measurement_probabilities(qubit_indices=list(range(d)))[0]

            assert_allclose(ps1, ps2)

    def test_measurement_probability_compatability2(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            d1 = np.random.randint(1, d + 1)
            qubit_indices = list(np.random.choice(d, size=d1, replace=False))

            state = random_state(d)
            reduced_rho = state.reduced_density_operator(qubit_indices)

            ps1 = state._measurement_probabilities(qubit_indices)[0]
            ps2 = reduced_rho._measurement_probabilities(qubit_indices=list(range(d1)))[0]

            assert_allclose(ps1, ps2)

    def test_measurement_probability_compatability3(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            M = np.random.randint(2, 10)
            d1 = np.random.randint(1, d + 1)
            qubit_indices = list(np.random.choice(d, size=d1, replace=False))

            rho = random_density_operator(d, M)
            reduced_rho = rho.reduced_density_operator(qubit_indices)

            ps1 = rho._measurement_probabilities(qubit_indices)[0]
            ps2 = reduced_rho._measurement_probabilities(qubit_indices=list(range(d1)))[0]

            assert_allclose(ps1, ps2)

    def test_measurement_probability_compatability4(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            M = np.random.randint(2, 10)

            rho = random_density_operator(d, M)
            purified_state = rho.purify()

            ps1 = rho._measurement_probabilities(qubit_indices=list(range(d)))[0]
            ps2 = purified_state._measurement_probabilities(qubit_indices=list(range(d)))[0]

            assert_allclose(ps1, ps2)

    def test_measurement_probability_compatability5(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            pure_state = random_unitary_operator(d)(qc.zeros(d))

            rho = pure_state.density_operator()
            purified_state = rho.purify()

            ps1 = pure_state._measurement_probabilities(qubit_indices=list(range(d)))[0]
            ps2 = purified_state._measurement_probabilities(qubit_indices=list(range(d)))[0]

            assert_allclose(ps1, ps2)

    def test_measurement_probability_compatability6(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            state = random_state(d)
            rho = state.density_operator()

            ps1 = state.probabilities
            ps2 = rho.probabilities

            assert_allclose(ps1, ps2)

    def test_measurement_probability_compatability7(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            d1 = np.random.randint(1, d + 1)
            M = np.random.randint(2, 10)
            qubit_indices = list(np.random.choice(d, size=d1, replace=False))

            rho = random_density_operator(d, M)
            reduced_rho = rho.reduced_density_operator(qubit_indices)

            ps1 = reduced_rho.probabilities.reshape(2**d1)
            ps2 = rho._measurement_probabilities(qubit_indices)[0]

            assert_allclose(ps1, ps2)

    def test_compare_reduction_methods(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(2, 7)
            d1 = np.random.randint(1, d + 1)
            M = np.random.randint(2, 10)
            retain_indices = list(np.random.choice(d, size=d1, replace=False))

            rho = random_density_operator(d, M)

            t1 = rho._reduced_tensor(retain_indices)

            # Old method
            traced_indices = list(
                set(range(rho.rank // 2)) - set(retain_indices)
            )
            t2 = rho._permuted_tensor(traced_indices + retain_indices)
            for _ in range(len(traced_indices)):
                t2 = np.sum(t2[[0, 1], [0, 1], ...], axis=0)

            assert_allclose(t1, t2)

    def test_purifying_pure_state(self):
        num_tests = 10

        for _ in range(num_tests):
            d = np.random.randint(1, 6)
            state = random_unitary_operator(d)(qc.zeros(d))
            rho = qc.DensityOperator.from_ensemble([state], [1.])

            purified_state = rho.purify()
            reduced_rho = purified_state.reduced_density_operator(qubit_indices=list(range(d)))

            assert_allclose(rho._t, reduced_rho._t)


class FastMeasurementTests(unittest.TestCase):

    def test_bitstring_measurement(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            bits = tuple(np.random.choice([0, 1], size=d, replace=True))
            state = qc.bitstring(*bits)
            measurement = state.measure()

            self.assertEqual(bits, measurement)

    def test_repeated_measurement_same(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            state = random_state(d)
            measurement1 = state.measure(remove=False)
            measurement2 = state.measure(remove=False)

            self.assertEqual(measurement1, measurement2)

    def test_repeated_single_qubit_measurement_same1(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            state = random_state(d)
            qubit_to_measure = int(np.random.randint(d))

            measurement1 = state.measure(remove=False)[qubit_to_measure]
            measurement2 = state.measure(qubit_indices=qubit_to_measure, remove=False)

            self.assertEqual(measurement1, measurement2)

    def test_repeated_single_qubit_measurement_same2(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            state = random_state(d)
            qubit_to_measure = int(np.random.randint(d))

            measurement1 = state.measure(qubit_indices=qubit_to_measure, remove=False)
            measurement2 = state.measure(qubit_indices=qubit_to_measure, remove=False)

            self.assertEqual(measurement1, measurement2)

    def test_U_f_basis_measurement(self):
        num_tests = 10

        for test_i in range(num_tests):
            d = np.random.randint(1, 8)
            f = random_boolean_function(d)
            U = qc.U_f(f, d=d+1)
            bits = tuple(np.random.choice([0, 1], size=d, replace=True))
            input_qubits = qc.bitstring(*bits)
            ans_qubit = qc.zeros()
            state = input_qubits * ans_qubit
            state = U(state)

            answer = f(*bits)
            measured_ans = state.measure(qubit_indices=d)

            self.assertEqual(answer, measured_ans)


sys.path.append('../examples')
from itertools import product
from deutsch_algorithm import deutsch_algorithm
import deutsch_jozsa_algorithm as dj_algorithm
from quantum_teleportation import quantum_teleportation
from superdense_coding import superdense_coding
import produce_bell_states
import quantum_parallelism
from phase_estimation import phase_estimation
import grover_algorithm

def deutsch_function(L):
    return lambda x: L[x]

class TestExamples(unittest.TestCase):
    """
    Test that the examples work as expected.
    """

    def test_deutsch_algorithm_example(self):
        for i, j in product([0, 1], repeat=2):
            f = deutsch_function([i, j])
            measurement = deutsch_algorithm(f)
            parity = int(i!=j)
            self.assertEqual(measurement, parity)

    def test_deutsch_jozsa_algorithm_example(self):
        num_tests = 10
        for problem_type in ['constant', 'balanced']:
            for test_i in range(num_tests):
                d = np.random.randint(3, 8)
                f = dj_algorithm.construct_problem(d, problem_type)
                measurements = dj_algorithm.deutsch_jozsa_algorithm(d, f)

                if problem_type == 'constant':
                    self.assertTrue(not(any(measurements)))
                else:
                    self.assertTrue(any(measurements))

    def test_grover_algorithm_example(self):
        num_tests = 3

        for test_i in range(num_tests):
            d = np.random.randint(8, 11)
            f = grover_algorithm.construct_problem(d)
            measurements = grover_algorithm.grover_algorithm(d, f)
            self.assertTrue(f(*measurements))

    def test_quantum_teleportation_example(self):
        num_tests = 10
        for test_i in range(num_tests):
            alice_state = random_state(d=1)
            bob_state = quantum_teleportation(alice_state)
            diff = max_absolute_difference(alice_state, bob_state)
            self.assertLess(diff, epsilon)

    def test_superdense_coding_example(self):
        num_tests = 10
        for test_i in range(num_tests):
            for bit_1, bit_2 in product([0, 1], repeat=2):
                measurements = superdense_coding(bit_1, bit_2)
                self.assertEqual(bit_1, measurements[0])
                self.assertEqual(bit_2, measurements[1])

    def test_bell_state_example(self):
        for x, y in product([0, 1], repeat=2):
            state1 = qc.bell_state(x, y)
            state2 = produce_bell_states.bell_state(x, y)
            diff = max_absolute_difference(state1, state2)
            self.assertLess(diff, epsilon)

    def test_quantum_parallelism_example(self):
        """
        This test currently just makes sure the example runs
        """

        f = quantum_parallelism.construct_problem()
        quantum_parallelism.quantum_parallelism(f)

    # TODO check error estimation calculation
    def test_phase_estimation_example(self):
        t = 7
        num_trials = 10
        failures = np.zeros(t)

        # per-bit failure rates are pretty certain to be less than these
        ts = np.array(range(t, 0, -1))
        epsilons = 1 / (2 * np.exp(ts) - 2)
        limits = np.array([binom(num_trials, e).ppf(0.99999) for e in epsilons])

        for trial_i in range(num_trials):

            phi, phi_estimate = phase_estimation(d=1, t=t)

            phi_b = [int(i) for i in bin(int(phi * 2**t))[2:]]
            est_b = [int(i) for i in bin(int(phi_estimate * 2**t))[2:]]
            if len(phi_b) < t:
                phi_b = (t-len(phi_b))*[0] + phi_b
            if len(est_b) < t:
                est_b = (t-len(est_b))*[0] + est_b
            phi_b = np.array(phi_b)
            est_b = np.array(est_b)
            failures += (phi_b != est_b)

        self.assertEqual(np.sum(failures > limits), 0)

if __name__ == '__main__':
    unittest.main()
