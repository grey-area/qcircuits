import qcircuits as qc
import numpy as np


# Superdense Coding: transmitting a qubit to transport two classical bits
# Alice and Bob have previously prepared a Bell state, and have since
# physically separated the qubits.
# Alice has two classical bits she wants to transmit to Bob.
# She manipulates her half of the Bell state depending on the values of those bits,
# then transmits her qubit to Bob, who then measures the system.


def superdense_coding(bit_1, bit_2):
    # Get operators we will need
    CNOT = qc.CNOT()
    H = qc.Hadamard()
    X = qc.PauliX()
    Z = qc.PauliZ()

    # The prepared, shared Bell state
    # Initially, half is in Alice's possession, and half in Bob's
    phi = qc.bell_state(0, 0)

    # Alice manipulates her qubit
    if bit_2:
        phi = X(phi, qubit_indices=[0])
    if bit_1:
        phi = Z(phi, qubit_indices=[0])

    # Bob decodes the two bits
    phi = CNOT(phi)
    phi = H(phi, qubit_indices=[0])
    measurements = phi.measure()
    return measurements


if __name__ == '__main__':
    # Alice's classical bits she wants to transmit
    bit_1, bit_2 = np.random.randint(0, 2, size=2)
    
    # Bob's measurements
    measurements = superdense_coding(bit_1, bit_2)
    
    print("Alice's initial bits:\t{}, {}".format(bit_1, bit_2))
    print("Bob's measurements:\t{}, {}".format(measurements[0], measurements[1]))
