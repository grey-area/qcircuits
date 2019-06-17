=========
QCircuits
=========

Full documentation at `www.awebb.info/qcircuits/index.html <http://www.awebb.info/qcircuits/index.html>`_.

.. inclusion-marker0-do-not-remove

QCircuits is a Python package for simulating quantum computers based on the
`quantum circuit model <https://en.wikipedia.org/wiki/Quantum_circuit>`_.
It has been designed to have a simple, lightweight interface and be
easy to use, particularly for those new to quantum computing.


.. inclusion-marker1-do-not-remove

Installation
============

Install with pip:

``pip install qcircuits``

.. inclusion-marker15-do-not-remove

or from the source available here.

.. inclusion-marker16-do-not-remove

Example usage: quantum teleportation
====================================

.. inclusion-marker2-do-not-remove

Quantum circuit:

.. image:: http://www.awebb.info/qcircuits/_images/teleport.png
    :scale: 40%

Code::

    import qcircuits as qc

    # Instantiating the operators we will need
    CNOT = qc.CNOT()
    H = qc.Hadamard()
    X = qc.PauliX()
    Z = qc.PauliZ()

    # Alice's hidden state, that she wishes to transport to Bob.
    alice = qc.qubit(theta=1, phi=1, global_phase=0.2)

    # A previously prepared Bell state, with one qubit owned by
    # alice, and another by Bob, now physically separated.
    bell_state = qc.bell_state(0, 0)

    # The state vector for the whole system.
    phi = alice * bell_state

    # Alice applies a CNOT gate to her two qubit, and then
    # a Hadamard gate to her private qubit.
    phi = CNOT(phi, qubit_indices=[0, 1])
    phi = H(phi, qubit_indices=[0])

    # Alice measures the first two bits, and transmits the classical
    # bits to Bob.
    # The only uncollapsed part of the state vector is Bob's.
    M1, M2 = phi.measure(qubit_indices=[0, 1], remove=True)

    # Apply X and/or Z gates to third qubit depending on measurements
    if M2:
        print('First bit 1, applying X\n')
        phi = X(phi)
    if M1:
        print('Second bit 1, applying Z\n')
        phi = Z(phi)

    print('Original state:', alice)
    print('\nTeleported state:', phi)

.. inclusion-marker3-do-not-remove

