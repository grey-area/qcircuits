import qcircuits as qc
import numpy as np


# Deutsch-Jorza Algorithhm:
# We are presented with a Boolean function that is either constant or
# balanced (i.e., 0 for half of inputs, 1 for the other half).
# We make use of interference to determine whether the function is constant
# or balanced in a single function evaluation.


# Construct a Boolean function that is constant or balanced
def construct_problem(d=1):
    num_inputs = 2**d
    answers = np.zeros(num_inputs, dtype=np.int32)

    if np.random.random() < 0.5: # function is constant
        problem_type = 'constant'
        answers[:] = int(np.random.random() < 0.5)
    else: # function is balanced
        problem_type = 'balanced'
        indices = np.random.choice(num_inputs, size=num_inputs//2, replace=False)
        answers[indices] = 1

    def f(*bits):
        index = sum(v * 2**i for i, v in enumerate(bits))

        return answers[index]

    return f, problem_type


if __name__ == '__main__':
    d = 10
    f, problem_type = construct_problem(d=d)
    U_f = qc.U_f(f, d=d+1)

    H_d = qc.Hadamard(d)
    H = qc.Hadamard()

    phi = qc.zeros(d) * qc.ones(1)
    phi = (H_d * H)(phi)
    phi = U_f(phi)
    phi = H_d(phi, qubit_indices=range(d))

    bits = []
    for d_i in range(d):
        bit, phi = phi.measure(qubit_index=0)
        bits.append(bit)

    print('Problem type: {}'.format(problem_type))
    print('Measurement: {}'.format(bits))
    print('Observed all zeros: {}'.format(not any(bits)))
