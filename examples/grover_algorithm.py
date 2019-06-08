import qcircuits as qc
import numpy as np
import random


# Grover's algorithm (search)
# Given a boolean function f, Grover's algorithm finds an x such that
# f(x) = 1.
# If there are N values of x, and M possible solutions, it requires
# O(sqrt(N/M)) time.
# Here, we construct a search problem with 1 solution amongst 1024
# possible answers, and find the solution with 25 applications of
# the Grover iteration operator.


# Construct a Boolean function that is 1 in exactly one place
def construct_problem(d=10):
    num_inputs = 2**d
    answers = np.zeros(num_inputs, dtype=np.int32)
    answers[np.random.randint(0, num_inputs)] = 1

    def f(*bits):
        index = sum(v * 2**i for i, v in enumerate(bits))

        return answers[index]

    return f


def grover_algorithm(d, f):
    # The operators we will need
    Oracle = qc.U_f(f, d=d+1)
    H_d = qc.Hadamard(d)
    H = qc.Hadamard()
    N = 2**d
    zero_projector = np.zeros((N, N))
    zero_projector[0, 0] = 1
    Inversion = H_d((2 * qc.Operator.from_matrix(zero_projector) - qc.Identity(d))(H_d))
    Grover = Inversion(Oracle, qubit_indices=range(d))

    # Initial state
    state = qc.zeros(d) * qc.ones(1)
    state = (H_d * H)(state)

    # Number of Grover iterations
    angle_to_rotate = np.arccos(np.sqrt(1 / N))
    rotation_angle = 2 * np.arcsin(np.sqrt(1 / N))
    iterations = int(round(angle_to_rotate / rotation_angle))
    for i in range(iterations):
        state = Grover(state)

    measurements = state.measure(qubit_indices=range(d))
    return measurements


if __name__ == '__main__':
    d = 10

    f = construct_problem(d)
    bits = grover_algorithm(d, f)

    print('Measurement: {}'.format(bits))
    print('Evaluate f at measurement: {}'.format(f(*bits)))