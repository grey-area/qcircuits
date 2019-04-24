.. _tutorial_page:

Tutorial
========

QCircuits is a library for simulating small-scale quantum circuits.
Its primary classes are :py:class:`.State`, representing the (quantum) state
of the computer, a unit vector in a
complex vector space, and :py:class:`.Operator`, representing quantum gates,
i.e. unitary operators,
on those vector spaces.

Specifically, QCircuits' states are for :math:`d`-qubit systems, and so they
are unit vectors in :math:`2^d`-dimensional complex vector spaces.

In the following, it is assumed that QCircuits has been imported with:

.. code-block:: python

    >>> import qcircuits as qc

States
------

This section introduces some of the ways in which states can be prepared
in QCircuits.

States in an all-zero or all-one computational basis state may be prepared
with the :py:func:`.zeros` and :py:func:`.ones` functions.
E.g., to prepare the state :math:`|\phi⟩ = |000⟩`:

.. code-block:: python

    >>> phi = qc.zeros(3)

And to prepare the state :math:`|\phi⟩ = |11⟩`:

.. code-block:: python

    >>> phi = qc.ones(2)

The :py:func:`.bitstring` function allows one to prepare a state in
an arbitrary computational basis state. E.g., to prepare the state 
:math:`|\phi⟩ = |01001⟩`:

.. code-block:: python

    >>> phi = qc.bitstring(0, 1, 0, 0, 1)

A single qubit may be prepared with the :py:func:`.qubit` function.
A qubit :math:`|\phi⟩ = \alpha |0⟩ + \beta |1⟩` may be prepared by providing
:math:`\alpha` and :math:`\beta` such that
:math:`\lvert\alpha\rvert^2 + \lvert\alpha\rvert^2 = 1`, e.g.,

.. code-block:: python

    >>> phi = qc.qubit(alpha=0.3, beta=0.7)

Alternatively, a qubit can be prepared in the state
:math:`e^{i \omega} \big( \cos(\theta/2) |0⟩ + e^{i \phi} \sin(\theta/2) |1⟩ \big)`,
where :math:`\omega` is the global phase of the state,
using the same function, e.g.,

.. code-block:: python

    >>> phi = qc.qubit(theta=math.pi, phi=0, global_phase=math.pi/2)

The four Bell states can be prepared using the :py:func:`.bell_state` function.
This takes two binary arguments x and y, and produces the Bell state
:math:`|\beta_{xy}⟩ = \big( |0, y⟩ + (-1)^x |1, 1-y⟩ \big)/\sqrt{2}`. E.g.,
the Bell state :math:`|\beta_{00}⟩ = \frac{|00⟩ + |11⟩}{\sqrt{2}}`
can be prepared:

.. code-block:: python

    >>> beta = qc.bell_state(0, 0)

The :py:func:`.positive_superposition` function may be used to prepare
a d-qubit state in the positive equal superposition of the computational
states. E.g., to construct the 2-qubit state 
:math:`|\phi⟩ = \big(|00⟩ + |01⟩ + |10⟩ + |11⟩ \big) / 2`:

.. code-block:: python

    >>> phi = qc.positive_superposition(d=2)

.. TODO: state arithmetic

.. TODO: qubit permutation

States can also be prepared by applying operators to states or taking the
tensor product of states, each of which is described in later sections.


How are States Represented?
---------------------------

Internally, QCircuits encodes a d-qubit state with an array of shape
(2, 2, ..., 2), with d axes in total, representing a tensor with
d contravariant indices. E.g., a 3-qubit state is represented by an array
of shape (2, 2, 2), and indexing into this array with indices i, j, k
gets the probability amplitude for the computational basis vector 
:math:`|ijk⟩`. The shape and the rank (number of axes) can be accessed
with the :py:attr:`.State.shape` and :py:attr:`.State.rank` properties.

.. code-block:: python

    >>> phi = qc.zeros(3)
    >>> print(phi)
    1-qubit state. Tensor:
    [[[1.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]

     [[0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]]

    >>> print(phi.shape)
    (2, 2, 2)

    >>> print(phi.rank)
    3

    >>> print(phi[0, 0, 0])
    (1+0j)

    >>> print(phi[0, 0, 1])
    0j

A d-qubit state can be constructed by providing this array.
E.g., a 3-qubit state can be constructed by providing a (2, 2, 2)
shape array:

.. code-block:: python

    >>> phi = qc.State([[[1., 0.],
    ...                  [0., 0.]],
    ...                 [[0., 0.],
    ...                  [0., 0.]]])

An alternative and common representation of a d-qubit state is as a column
vector of length :math:`2^d`. This column-vector representation can
be obtained with the :py:meth:`.State.to_column_vector` method:

.. code-block:: python

    >>> phi = qc.bitstring(0, 1, 0)
    >>> phi.to_column_vector()
    array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

States can be constructed from the column vector representation using
the :py:meth:`.State.from_column_vector` static method:

.. code-block:: python

    >>> phi = qc.State.from_column_vector(
    ...     [0., 0., 1., 0., 0., 0., 0., 0.]
    ... )


Operators
---------

.. TODO

.. TODO operator arithmetic

For a list of available operators, see :py:class:`.Operator`.


How are Operators Represented?
------------------------------

Internally, QCircuits encodes an operator for a d-qubit system with an array of shape
(2, 2, ..., 2), with 2d axes in total, representing a tensor with
d contravariant indices and d covariant indices.
E.g., an operator for a 2-qubit system is represented by an array
of shape (2, 2, 2, 2).
The shape and the rank (number of axes) can be accessed
with the :py:attr:`.Operator.shape` and :py:attr:`.Operator.rank` properties.

.. code-block:: python

    >>> H = qc.Hadamard()
    >>> print(H)

    Operator for 1-qubit state space. Tensor:
    [[ 0.70710678+0.j  0.70710678+0.j]
     [ 0.70710678+0.j -0.70710678+0.j]]

    >>> print(H.shape)
    (2, 2)

    >>> print(H.rank)
    2

We use the convention that the covariant and contravariant indices
alternate.
The result is that indexing into the array representing operator U with indices i, j, k, ...
in the odd-numbered places gets the state the computational basis vector
:math:`|ijk\ldots⟩` is taken to by the operator.

.. code-block:: python

    >>> print(U[:, 0])
    [ 0.53114041-0.31105474j -0.69143236-0.37822758j]

    >>> phi = qc.zeros(1)
    >>> print(U(phi))
    [ 0.53114041-0.31105474j -0.69143236-0.37822758j]

    >>> print(V[:, 1, :, 0])
    array([[-0.66947579+0.19664594j, -0.37841556-0.24010317j],
           [ 0.30464249-0.40638463j,  0.16243857-0.16716121j]])

    >>> phi = qc.bitstring(1, 0)
    >>> print(V(phi))
    array([[-0.66947579+0.19664594j, -0.37841556-0.24010317j],
           [ 0.30464249-0.40638463j,  0.16243857-0.16716121j]])

An operator can be constructed by providing an array of the appropriate shape.
E.g., the two qubit Hadamard gate :math:`H\otimes H` can be constructed
by providing the (2, 2, 2, 2)-shape array:

.. code-block:: python

    >>> H = qc.Operator([[[[ 0.5,  0.5],
    ...                    [ 0.5, -0.5]],
    ...                   [[ 0.5,  0.5],
    ...                    [ 0.5, -0.5]]],
    ...          
    ...                  [[[ 0.5,  0.5],
    ...                    [ 0.5, -0.5]],
    ...                   [[-0.5, -0.5],
    ...                    [-0.5,  0.5]]]])

An alterantive and common representation of d-qubit operators is as a
:math:`2^d \times 2^d` matrix. This matrix representation can be accessed
with the :py:meth:`.Operator.to_matrix` method. E.g., for the two-qubit
Hadamard gate:

.. code-block:: python

    >>> print(H.to_matrix())
    [[ 0.5+0.j  0.5+0.j  0.5+0.j  0.5+0.j]
     [ 0.5+0.j -0.5+0.j  0.5+0.j -0.5+0.j]
     [ 0.5+0.j  0.5+0.j -0.5+0.j -0.5+0.j]
     [ 0.5+0.j -0.5+0.j -0.5+0.j  0.5+0.j]]

Operators can be constructed from this matrix representation using the 
:py:meth:`.Operator.from_matrix` static method:

.. code-block:: python

    >>> H = qc.Operator(
    ...     [[ 0.5,  0.5,  0.5,  0.5],
    ...      [ 0.5, -0.5,  0.5, -0.5],
    ...      [ 0.5,  0.5, -0.5, -0.5],
    ...      [ 0.5, -0.5, -0.5,  0.5]]
    ... )


Tensor Products
---------------

.. TODO

The infix multiplication operator * can be used to take the tensor product of
states or operators. E.g., the state :math:`|\psi⟩ \otimes |\phi⟩`:

.. code-block:: python

    >>> psi * phi

Where :math:`A` and :math:`B` are operators, the operator
:math:`A\otimes B`:

.. code-block:: python

    >>> A * B


Applying Operators to States
----------------------------

.. TODO

Composing Operators
-------------------

.. TODO

Measurement
-----------

.. TODO


Entanglement / Schmidt Number
-----------------------------

.. TODO