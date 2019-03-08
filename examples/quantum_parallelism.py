import qcircuits as qc
import numpy as np


# Example of quantum parallelism


# Construct a Boolean function
def construct_problem():
    answers = np.random.randint(0, 2, size=2)

    def f(bit):
        return answers[bit]

    return f


if __name__ == '__main__':
    f = construct_problem()
    H = qc.H()

    phi = qc.zeros(2)
    phi = H(phi, qubit_indices=[0])
    phi = qc.U_f(f, d=2)(phi)
