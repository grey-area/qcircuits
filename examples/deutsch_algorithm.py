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


if __name__ == '__main__':
    f = construct_problem()
    U_f = qc.U_f(f, d=2)
    H = qc.Hadamard()

    phi = H(qc.zeros()) * H(qc.ones())
    phi = U_f(phi)
    phi = H(phi, qubit_indices=[0])

    measurement, phi = phi.measure(qubit_index=0)

    print(f'f(0): {f(0)}, f(1): {f(1)}')
    print(f'f(0) == f(1): {f(0) == f(1)}')
    print(f'Measurement: {measurement}')
