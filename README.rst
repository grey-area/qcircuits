=========
QCircuits
=========

QCircuits is a lightweight package for simulating the operation of
small-scale quantum computers, based on the quantum circuit model.
It uses rank d and rank 2d tensors to represent state vectors for and operators on d-qubit systems,
rather than using straight vectors and matrices
produces by Kronecker products.

Example usage, performing quantum teleportation::

    import qcircuits as qc

    # Instantiating the operators we will need
    CNOT = qc.CNOT()
    H = qc.Hadamard()
    X = qc.PauliX()
    Z = qc.PauliZ()

    # Alice's hidden state, and the shared Bell state
    alice = qc.qubit(theta=1, phi=1, global_phase=0.2)
    bell_state = qc.bell_state(0, 0)
    phi = alice * bell_state

    phi = CNOT(phi, qubit_indices=[0, 1])
    phi = H(phi, qubit_indices=[0])

    # Measure the first two bits
    M1, phi = phi.measure(qubit_index=0)
    M2, bob = phi.measure(qubit_index=0)

    # Apply gates based on the classical bits Alice
    # sends to Bob
    if M2:
        bob = X(bob)
    if M1:
        bob = Z(bob)

    print('Original state:', alice)
    print('\nTeleported state:', bob)
