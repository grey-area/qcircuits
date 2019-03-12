import qcircuits as qc
import numpy as np
import random


# Deutsch-Jorza Algorithhm:
# We are presented with a Boolean function that is either constant or
# balanced (i.e., 0 for half of inputs, 1 for the other half).
# We make use of interference to determine whether the function is constant
# or balanced in a single function evaluation.


# Construct a Boolean function that is constant or balanced
def construct_problem(d=1, problem_type='constant'):
    num_inputs = 2**d
    answers = np.zeros(num_inputs, dtype=np.int32)

    if problem_type == 'constant':
        answers[:] = int(np.random.random() < 0.5)
    else: # function is balanced
        indices = np.random.choice(num_inputs, size=num_inputs//2, replace=False)
        answers[indices] = 1

    def f(*bits):
        index = sum(v * 2**i for i, v in enumerate(bits))

        return answers[index]

    return f


def deutsch_jorza_algorithm(d, f):
    # The operators we will need
    U_f = qc.U_f(f, d=d+1)
    H_d = qc.Hadamard(d)
    H = qc.Hadamard()

    state = qc.zeros(d) * qc.ones(1)
    state = (H_d * H)(state)
    state = U_f(state)
    state = H_d(state, qubit_indices=range(d))

    measurements = state.measure(qubit_indices=range(d))
    return measurements


if __name__ == '__main__':
    d = 10
    problem_type = random.choice(['constant', 'balanced'])

    f = construct_problem(d, problem_type)
    measurements = deutsch_jorza_algorithm(d, f)

    print('Problem type: {}'.format(problem_type))
    print('Measurement: {}'.format(measurements))
    print('Observed all zeros: {}'.format(not any(measurements)))
