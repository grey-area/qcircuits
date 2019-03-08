import qcircuits as qc
from itertools import product


# Creates each of the four Bell states


def bell_state(x, y):
    H = qc.Hadamard()
    CNOT = qc.CNOT()

    phi = qc.bitstring(x, y)
    phi = H(phi, qubit_indices=[0])

    return CNOT(phi)


if __name__ == '__main__':

    for x, y in product([0, 1], repeat=2):

        print(f'\nInput: {x} {y}')
        print('Bell state:')
        print(bell_state(x, y))
