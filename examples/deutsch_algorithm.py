import qcircuits as qc
import numpy as np


# Deutsch's Algorithhm:
# We use interference to determine if f(0) = f(1) using a single function evaluation.


# Construct a Boolean function that is constant or balanced
def construct_problem():
    answers = np.random.randint(0, 2, size=2)

    def f(bit):
        return answers[bit]

    return f


def deutsch_algorithm(f):
    U_f = qc.U_f(f, d=2)
    H = qc.Hadamard()

    phi = H(qc.zeros()) * H(qc.ones())
    phi = U_f(phi)
    phi = H(phi, qubit_indices=[0])

    measurement = phi.measure(qubit_indices=0)
    return measurement


if __name__ == '__main__':
    f = construct_problem()
    parity = f(0) == f(1)

    measurement = deutsch_algorithm(f)

    print('f(0): {}, f(1): {}'.format(f(0), f(1)))
    print('f(0) == f(1): {}'.format(parity))
    print('Measurement: {}'.format(measurement))
