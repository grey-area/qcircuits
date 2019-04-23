.. _tutorial_page:

Tutorial
========

QCircuits is a library for simulating small-scale quantum circuits.
Its primary classes are :py:class:`.State`, representing a unit vector in a
complex vector space, and :py:class:`.Operator`, representing unitary operators
on those vector spaces.

Specifically, QCircuits' states are for :math:`d`-qubit systems, and so they
are unit vectors in :math:`2^d`-dimensional complex vector spaces.

In the following, it is assumed that QCircuits has been imported with:

.. code-block:: python

    >>> import qcircuits as qc

Constructing states
-------------------

This section introduces some of the ways in which states can be prepared
in QCircuits.

States in an all-zero or all-one computational basis state may be prepared
with the :py:func:`.zeros` and :py:func:`.ones` functions.
E.g., to prepare the state :math:`|000⟩`:

.. code-block:: python

    >>> state = qc.zeros(3)

And to prepare the state :math:`|11⟩`:

.. code-block:: python

    >>> state = qc.ones(2)

Internally, QCircuits encodes a d-qubit state with an array of shape
(2, 2, ..., 2), with d axes in total, representing a tensor with
d contravariant indices. E.g., a 3-qubit state is represented by an array
of shape (2, 2, 2), and indexing into this array with indices i, j, k
gets the probability amplitude for the computational basis vector 
:math:`|ijk⟩`

.. code-block:: python

    >>> state = qc.zeros(3)
    >>> print(state)
    1-qubit state. Tensor:
    [[[1.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]

     [[0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]]

    >>> print(state[0, 0, 0])
    (1+0j)

    >>> print(state[0, 0, 1])
    0j

The :py:func:`.bitstring` function allows one to prepare a state in
an arbitrary computational basis state. E.g., to prepare the state 
:math:`|0010⟩`:

.. code-block:: python

    >>> state = qc.bitstring(0, 0, 1, 0)
    
A single qubit may be prepared with the :py:func:`.qubit` function.
:math:`\alpha |0⟩ + \beta |1⟩`

:math:`e^{i \omega} ( \cos \theta |0⟩ + e^{i \phi} \sin \theta |1⟩ )`

Constructing operators
----------------------