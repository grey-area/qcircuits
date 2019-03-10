=========
QCircuits
=========

Full documentation at `www.awebb.info/qcircuits/index.html <http://www.awebb.info/qcircuits/index.html>`_.

.. inclusion-marker0-do-not-remove

QCircuits is a lightweight Python package for simulating the operation of
small-scale quantum computers, based on the
`quantum circuit model <https://en.wikipedia.org/wiki/Quantum_circuit>`_.
It uses type (`d`, 0) and type (`d`, `d`) tensors to represent state vectors
for and operators on `d`-qubit systems,
rather than using straight vectors and matrices
produced by Kronecker products, as is more typical.

.. inclusion-marker1-do-not-remove

Installation
============

Install with pip:

``pip install qcircuits``

.. inclusion-marker15-do-not-remove

or from the source available here.

.. inclusion-marker16-do-not-remove

Example usage
=============

.. inclusion-marker2-do-not-remove

Quantum teleportation example:

.. literalinclude:: examples/quantum_teleportation.py


Details
=======
