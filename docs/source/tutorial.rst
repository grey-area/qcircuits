.. _tutorial_page:

Tutorial
========

QCircuits is a library for simulating quantum circuits.
Its primary classes are :py:class:`.State`, representing the (quantum) state
of the computer, a unit vector in a
complex vector space, and :py:class:`.Operator`, representing quantum gates,
i.e. unitary operators
on those vector spaces.
This tutorial describes how to use QCircuits to prepare operators and states,
apply operators to states, measure states, etc., in order to implement
quantum algorithms.

First, install QCircuits with pip:

``pip install qcircuits``

In the following, it is assumed that QCircuits has been imported with:

.. code-block:: python

    >>> import qcircuits as qc

States
======

States in QCircuits represent the state of a :math:`d`-qubit quantum computer.
This section introduces some of the ways in which states can be prepared
in QCircuits.

States in an all-zero or all-one computational basis state may be prepared
with the :py:func:`.zeros` and :py:func:`.ones` functions.
E.g., to prepare the state :math:`|\phi⟩ = |000⟩`:

.. code-block:: python

    >>> phi = qc.zeros(3)  # the state |000⟩

And to prepare the state :math:`|\phi⟩ = |11⟩`:

.. code-block:: python

    >>> phi = qc.ones(2)  # the state |11⟩

The :py:func:`.bitstring` function allows one to prepare a state in
an arbitrary computational basis state. E.g., to prepare the state
:math:`|\phi⟩ = |01001⟩`:

.. code-block:: python

    >>> phi = qc.bitstring(0, 1, 0, 0, 1)  # the state |01001⟩

A single qubit may be prepared with the :py:func:`.qubit` function.
A qubit :math:`|\phi⟩ = \alpha |0⟩ + \beta |1⟩` may be prepared by providing
:math:`\alpha` and :math:`\beta` such that
:math:`\lvert\alpha\rvert^2 + \lvert\beta\rvert^2 = 1`, e.g.,

.. code-block:: python

    >>> phi = qc.qubit(alpha=0.5, beta=np.sqrt(3)/2)  # the state 0.5|0⟩ + 0.866|1⟩

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

    >>> beta = qc.bell_state(0, 0)  # the state (|0⟩ + |1⟩)/1.414

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
===========================

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
    3-qubit state. Tensor:
    [[[1.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]

     [[0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]]

    >>> print(phi.shape)
    (2, 2, 2)

    >>> print(phi.rank)
    3

    >>> print(phi[0, 0, 0])  # the probability amplitude of our state for |000⟩
    (1+0j)

    >>> print(phi[0, 0, 1])  # the probability amplitude of our state for |001⟩
    0j

A d-qubit state can be constructed by providing this array.
E.g., a 3-qubit state can be constructed by providing a (2, 2, 2)
shape array:

.. code-block:: python

    >>> phi = qc.State([[[1., 0.],  # the state |000⟩
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

    >>> phi = qc.State.from_column_vector(   # the state |010⟩
    ...     [0., 0., 1., 0., 0., 0., 0., 0.]
    ... )

Note that while QCircuits allows the user to extract the tensor data from a state,
i.e., observe a state in full, rather than just taking a measurement,
in true quantum computation this is not possible.


Operators
=========

Operators in QCircuits represent quantum 'gates' for :math:`d`-qubit
quantum computers, i.e., unitary linear operators on a :math:`2^d` dimensional
complex vector space.

This section describes some of the built-in operators that may be used.

The :py:func:`.PauliX`, :py:func:`.PauliY`, and :py:func:`.PauliZ` functions
return instances of the common X, Y, and Z gates. Here they are shown acting
on some computational basis vectors as expected:

.. code-block:: python

    >>> X = qc.PauliX()  # The X gate, or NOT gate
    >>> phi = qc.zeros(1)
    >>> result = X(phi)  # apply the X gate to the state |0⟩
    >>> print(result)    # the result is the state |1⟩
    1-qubit state. Tensor:
    [0.+0.j 1.+0.j]

.. code-block:: python

    >>> Y = qc.PauliY()
    >>> phi = qc.zeros(1)
    >>> result = Y(phi) # apply the Y gate to the state |0⟩
    >>> print(result)   # the result is the state i|1⟩
    1-qubit state. Tensor:
    [0.+0.j 0.+1.j]

.. code-block:: python

    >>> Z = qc.PauliZ()
    >>> result = Z(qc.zeros(1))  # apply the Z gate to the state |0⟩
    >>> print(result)  # the result is the state |0⟩
    1-qubit state. Tensor:
    [1.+0.j 0.+0.j]

    >>> result = Z(qc.ones(1))  # apply the Z gate to the state |1⟩
    >>> print(result)  # the result is the state -|1⟩
    1-qubit state. Tensor:
    [-0.-0.j -1.-0.j]

We have seen in the above examples that operators are applied to states
by function application, i.e., U(v), where U is an operator and v a state.
Operator application will be described in more detail later in the tutorial.

An instance of the Hadamard gate can be obtained with the
:py:func:`.Hadamard` function. Here we see an example of applying the
Hadamard operator to the state :math:`|0⟩`:

.. code-block:: python

    >>> H = qc.Hadamard()
    >>> phi = qc.zeros(1)  # the state |0⟩
    >>> result = H(phi)    # the state (|0⟩ + |1⟩) / sqrt(2)
    >>> print(result)
    1-qubit state. Tensor:
    [0.70710678+0.j 0.70710678+0.j]

The above functions, which return instances of the the X, Y, Z, and Hadamard
gates, take an integer argument d. The returned d-qubit operator applies the gate in
question to each qubit independently. E.g., for the X gate, the returned operator is
:math:`X^{\otimes d}`.

.. code-block:: python

    >>> X = qc.PauliX(d=3)  # The 3-qubit operator applying X to each qubit
    >>> phi = qc.bitstring(0, 1, 0)  # the state |010⟩
    >>> result = X(phi)  # the state |101⟩
    >>> print(result)
    1-qubit state. Tensor:
    [[[0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]

     [[0.+0.j 1.+0.j]
      [0.+0.j 0.+0.j]]]

By default, dimensionality of the operator and the state it is applied to
must match. I.e., a d-qubit operator must be applied to a d-qubit state.
Ways of applying n-qubit operators to d-qubit states where n is less than d
will be discussed in the sections on tensor products and operator application
below.

The :py:func:`.CNOT` function returns an instance of the CNOT operator,
i.e., the 2-qubit operator that applies the X operator to the second (target) qubit
if the first (control) qubit is in state :math:`|1⟩`.

.. code-block:: python

    >>> CNOT = qc.CNOT()
    >>> print(CNOT(qc.bitstring(0, 0)))   # |00⟩ -> |00⟩
    2-qubit state. Tensor:
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> print(CNOT(qc.bitstring(0, 1)))   # |01⟩ -> |01⟩
    2-qubit state. Tensor:
    [[0.+0.j 1.+0.j]
     [0.+0.j 0.+0.j]]

    >>> print(CNOT(qc.bitstring(1, 0)))   # |10⟩ -> |11⟩
    2-qubit state. Tensor:
    [[0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j]]

    >>> print(CNOT(qc.bitstring(1, 1)))   # |11⟩ -> |10⟩
    2-qubit state. Tensor:
    [[0.+0.j 0.+0.j]
     [1.+0.j 0.+0.j]]

Often, we want to swap the role of the qubits, flipping the first qubit if the second
is set, or more generally, for a d-qubit state we may want to apply the 2-qubit CNOT
operator on any two qubits. How this is done is described in the section on
operator application below.

The :py:func:`.ControlledU` function takes a d-qubit operator as an argument,
and returns the (d+1)-qubit controlled-U operator: if the first qubit is set,
the operator U is applied to the following d qubits.

.. code-block:: python

    >>> phi0 = qc.bitstring(0, 0, 0)  # the 3-qubit state |000⟩
    >>> phi1 = qc.bitstring(1, 0, 0)  # the 3-qubit state |100⟩
    >>> H = qc.Hadamard(d=2)  # a 2-qubit operator, applying the Hadamard operator to each qubit
    >>> c_H = qc.ControlledU(H)  # a 3-qubit operator, applying the Hadamard operator
    ...                          # to qubits 2 and 3 if qubit 1 is set
    >>> print(c_H(phi0))  # the state is left unchanged
    3-qubit state. Tensor:
    [[[1.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]

     [[0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]]

    >>> print(c_H(phi1))  # the 2-qubit H operator is applied
    1-qubit state. Tensor:
    [[[0. +0.j 0. +0.j]
      [0. +0.j 0. +0.j]]

     [[0.5+0.j 0.5+0.j]
      [0.5+0.j 0.5+0.j]]]

The :py:func:`.U_f` function takes two arguments: a function f and an integer d.
The function f must be a boolean function of d-1 boolean arguments.
This returns a d-qubit operator whose action is to flip the last qubit
if the result of applying the boolean function to the first d-1 qubits is one.
An example of its use can be found in the Deutsch-Jozsa algorithm in the
:ref:`examples page<examples_page>`.

.. TODO operator arithmetic

For a full list of available operators, see :py:class:`.Operator`.


How are Operators Represented?
==============================

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

    >>> print(U[:, 0])  # Indexing in with 0 here gets the tensor representation of
    ...                 # the operator applied to the state |0⟩
    [ 0.53114041-0.31105474j -0.69143236-0.37822758j]

    >>> phi = qc.zeros(1)
    >>> print(U(phi))
    [ 0.53114041-0.31105474j -0.69143236-0.37822758j]

    >>> print(V[:, 1, :, 0])  # Indexing in with 1, 0 here gets the tensor representation
    ...                       # of the operator applied to the state |10⟩
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



Applying Operators to States
============================

Operators are applied to states as follows.

.. code-block:: python

    >>> H = qc.Hadamard()  # the single-qubit Hadamard gate
    >>> x = qc.zeros(1)    # a single qubit in the |0⟩ state
    >>> y = H(x)           # apply the Hadamard gate to the qubit to obtain an equal superposition state


When applying an operator to a state, a qubit_indices argument can be supplied in
order to permute the order of the state for the operator application.
For example, the :py:func:`.CNOT` gate flips the second bit if the first bit is set:

.. code-block:: python

    >>> CNOT = qc.CNOT()        # the CNOT gate
    >>> x = qc.bitstring(1, 1)  # the |11⟩ state
    >>> y = CNOT(x)             # the resulting |10⟩ state

Passing in the qubit_indices argument allows us to specify the roles of the two
qubits explicitly. For example, the following swaps the roles of the qubits during
operator application, so that the first qubit flips if the second is set.

.. code-block:: python

    >>> y = CNOT(x, qubit_indices=[1, 0])  # the resulting |01⟩ state

The qubit_indices argument can also be used to apply an :math:`m`-qubit operator to an
:math:`n`-qubit state where :math:`n>m`. For example, the following code applies a two qubit
Hadamard operator to the first and third qubits of a six qubit state.

.. code-block:: python

    >>> H = qc.Hadamard(d=2)   # the two-qubit Hadamard gate
    >>> x = qc.zeros(6)        # the six-qubit state |000000⟩ state
    >>> y = H(x, qubit_indices=[0, 2])  # resulting state |+0+000⟩

This is useful, as it is much more efficient than expanding the :math:`m`-qubit operator
to a :math:`n`-qubit operator by taking the tensor product with the identity and then applying the result.



Tensor Products
===============

If a quantum system :math:`A` is in state :math:`|\psi⟩`, and system
:math:`B` is in state :math:`|\phi⟩`, then the combined system
:math:`A\otimes B` is in state :math:`|\psi⟩ \otimes |\phi⟩`,
where :math:`\otimes` is the tensor product. If operator :math:`U`
is applied to system :math:`A` and operator V applied to system :math:`B`,
then this can be described by a single operator :math:`U\otimes V` applied
to system :math:`A\otimes B`.

This tensor product operation can be used via the infix multiplication
operator * in QCircuits to produce operators and states for larger systems
from operators and states for smaller systems:

.. code-block:: python

    >>> psi = qc.bitstring(0)   # the state |0⟩
    >>> phi = qc.bitstring(11)  # the state |11⟩
    >>> state = psi * phi       # the state |011⟩
    >>> print(state)
    3-qubit state. Tensor:
    [[[0.+0.j 0.+0.j]
      [0.+0.j 1.+0.j]]

     [[0.+0.j 0.+0.j]
      [0.+0.j 0.+0.j]]]

.. code-block:: python

    >>> X = qc.PauliX()  # a single qubit X operator
    >>> Z = qc.PauliZ(d=2)  # a 2-qubit operator applying Z independently to two qubits
    >>> U = X * Z  # a 3-qubit operator, applying X to the first qubit and Z to the second and third

One use of taking the tensor product of operators is applying smaller operators to larger states.
E.g., a 1-qubit operator may be applied to one of the qubits of a 2-qubit state by taking the tensor
product of the operator with the identity operator.

.. code-block:: python

    >>> H = qc.Hadamard()  # the 1-qubit Hadamard operator
    >>> I = qc.Identity()  # the 1-qubit identity operator
    >>> phi = qc.bitstring(0, 0)  # the state |00⟩
    >>> print((H * I)(phi))  # apply the Hadamard operator to the first qubit
    ...                      # resulting in state (|0⟩ + |1⟩)/sqrt(2) |0⟩
    2-qubit state. Tensor:
    [[0.70710678+0.j 0.        +0.j]
     [0.70710678+0.j 0.        +0.j]]

    >>> print((I * H)(phi))  # apply the Hadamard operator to the second qubit
    ...                      # resulting in state |0⟩ (|0⟩ + |1⟩)/sqrt(2)
    2-qubit state. Tensor:
    [[0.70710678+0.j 0.70710678+0.j]
     [0.        +0.j 0.        +0.j]]

Since a d-qubit operator is specified with :math:`2^{2d}` complex values,
while a d-qubit state is specified with :math:`2^d` complex values,
this method is not ideal when applying very small operators to very large states.
As an example, working with 30-qubit states is plausible on personal hardware,
requiring 16 GB of memory.
Expanding an operator to a 30-qubit operator to act on this state is not plausible,
as this would require 16 exabytes (1 million TB) of memory.
The following section describes an alternative way to apply smaller operators to larger
states.

One can use power notation to take the tensor product of an operator or state
with itself :math:`n` times, as in the following code that creates an :math:`n`-qubit
operator from a single-qubit operator:

.. code-block:: python

    >>> Op2 = Op**n  # creating an n-qubit operator from a single-qubit operator





Composing Operators
===================

Operators may be applied to other operators to produce new operators,
as linear operators are associative, i.e., :math:`A(B|\phi⟩) = (AB)|\phi⟩`.
For example, suppose we start with state :math:`|00⟩`, and wish to apply
the Hadamard gate to the first qubit, then apply the CNOT gate, resulting
in a Bell state. We can either apply these operators in sequence, or
we can first construct a single 2-qubit operator by composing the operators,
and apply the resulting operator to the state.

.. code-block:: python

    >>> H = qc.Hadamard()
    >>> I = qc.Identity()
    >>> CNOT = qc.CNOT()
    >>> phi = qc.bitstring(0, 0)

    >>> result1 = CNOT((H * I)(phi))  # these result in the same state
    >>> U = CNOT(H * I)  # this method produces an operator U that performs both operators
    >>> result2 = U(phi)  # the result is a Bell state

Suppose we wish to apply the inverse operator, taking a Bell state to the state
:math:`|00⟩`. First we apply the CNOT operator to the state, then we apply
the 1-qubit Hadamard operator to the first qubit.
Again, this can be done with operator composition:

.. code-block:: python

    >>> H = qc.Hadamard()
    >>> I = qc.Hadamard()
    >>> CNOT = qc.CNOT()
    >>> phi = qc.bell_state(0, 0)
    >>> result = (H * I)(CNOT)(phi)  # first H*I is applied to CNOT, composing the
    ...                              # operators. The result is applied to the state

In this case, though, where we are applying a smaller operator to a larger operator,
we can use the same interface as when applying a smaller operator to a larger state,
by specifying the qubits on which the operator acts:

.. code-block:: python

    >>> H = qc.Hadamard()
    >>> CNOT = qc.CNOT()
    >>> phi = qc.bell_state(0, 0)
    >>> U = H(CNOT, qubit_indices=[0])
    >>> result = U(phi)

.. TODO think of supplying qubit indices as wire permutation

Measurement
===========

Measurement in the computational basis may be performed on one or
more qubits of a state with the :py:meth:`.State.measure` method,
which returns the result of measurement.
Post-measurement, the state collapses to the computational basis
state corresponding to the result of the measurement of the measured
qubits. The :py:meth:`.State.measure` method has two arguments:
a list of indices specifying the qubits to be measured,
and a flag specifying whether the measured qubits are to be removed from the state.
If no indices are supplied, every qubit is measured.

E.g., the single-qubit state :math:`|0⟩` will yield the measurement 0
with certainty:

.. code-block:: python

    >>> qc.zeros(1).measure()
    (0,)

The 3-qubit state :math:`|111⟩` will yield measurement 1, 1 with certainty when the
first two qubits are measured. Post-measurement we are left with either a 3- or 1-qubit
state, depending on if we choose to remove the measured qubits from the state.

.. code-block:: python

    >>> phi = qc.ones(3)
    >>> phi.measure(qubit_indices=[0, 1], remove=False)
    (1,1)
    >>> phi.shape
    (2,2,2)

    >>> phi = qc.ones(3)
    >>> phi.measure(qubit_indices=[0, 1], remove=True)
    (1,1)
    >>> phi.shape  # if the measured qubits are removed, we are left with state |1⟩
    (2,)

Measuring the state :math:`(|0⟩ + |1⟩)/\sqrt{2}` yields 0 or 1 with equal probability.
Note that in the following code we must prepare a fresh state each time we measure
to see different measurement outcomes, because the measured state is left either
in the state :math:`|0⟩` or in the state :math:`|1⟩`.

.. code-block:: python

    >>> H = qc.Hadamard()
    >>> H(qc.zeros(1)).measure()
    (0,)

    >>> H(qc.zeros(1)).measure()
    (1,)

    >>> H(qc.zeros(1)).measure()
    (0,)




.. TODO warning about non-unit, non-unitary


Density Operators
=================

`Density operators <https://en.wikipedia.org/wiki/Density_matrix>`_ are a useful
way of representing mixed states, i.e., statistical ensembles of quantum states
that can arise when there is uncertainty about the actual quantum states that
are being manipulated. If a system is known to be in one of a collection of
states :math:`|\phi_i⟩`, each with probability :math:`p_i`, then the density
operator representation of this state is :math:`\rho = \sum_i p_i |\phi_i⟩⟨\phi_i|`.

A DensityOperator object can be created with the :py:meth:`.DensityOperator.from_ensemble`
static method, by supplying a list
of :py:class:`.State` objects and a list of probabilities, matching in length,
which must sum to one:

.. code-block:: python

    >>> state1 = qc.bitstring(0, 1, 0)  # the state |010⟩
    >>> state2 = qc.bitstring(1, 0, 0)  # the state |100⟩
    >>> mixed_state = qc.DensityOperator.from_ensemble(
    ...     [state1, state2],
    ...     ps=[0.3, 0.7]
    ... )  # the statistical ensemble known to be in states |010⟩ and |100⟩
    ...    # each with probability 1/2.

The collection of states must have matching rank. I.e., one cannot form a mixture
of a two qubit and a three qubit state. If the probability vector is omitted a
uniform mixture is assumed.

A DensityOperator can be created from a single pure state using the :py:meth:`.State.density_operator`
method:

.. code-block:: python

    >>> state = qc.bitstring(0, 0, 0)  # the state |000⟩
    >>> rho = state.density_operator()

Operators can be applied to mixed states in the same way that they are
applied to pure states:

.. code-block:: python

    >>> H = qc.Hadamard(d=3)  # the three-qubit Hadamard operator
    >>> state1 = qc.bitstring(0, 1, 0)  # the state |010⟩
    >>> state2 = qc.bitstring(1, 0, 0)  # the state |100⟩
    >>> rho = qc.DensityOperator.from_ensemble(
    ...     [state1, state2],
    ...     ps=[0.3, 0.7]
    ... )  # the statistical ensemble known to be in states |010⟩ and |100⟩
    ...    # each with probability 1/2.
    >>> result = H(rho)  # apply the operator to the state

Under the hood, this takes the density operator :math:`\rho` to
:math:`H \rho H^{\dagger}`, which is the resulting density operator if
the operator :math:`H` is applied to the state whose uncertainty is
being represented.

If the operator is for a lower-dimensional space than the density operator
represents, or if you want to permute the roles of the qubits in the
operator application, a qubit_indices argument can be supplied as when
applying operators to pure states:

.. code-block:: python

    >>> H = qc.Hadamard(d=2)  # the two-qubit Hadamard operator
    >>> state1 = qc.bitstring(0, 1, 0)  # the state |010⟩
    >>> state2 = qc.bitstring(1, 0, 0)  # the state |100⟩
    >>> rho = qc.DensityOperator.from_ensemble(
    ...     [state1, state2],
    ...     ps=[0.3, 0.7]
    ... )  # the statistical ensemble known to be in states |010⟩ and |100⟩
    ...    # each with probability 1/2.
    >>> result = H(rho, qubit_indices=[0, 2])  # apply the two-qubit operator to
    ...                                        # qubits 0 and 2


Measurement in the computational basis may be performed on one or
more qubits of a mixed state with the :py:meth:`.DensityOperator.measure` method,
which returns the result of measurement.

.. code-block:: python

    >>> state1 = qc.zeros(1)  # the state |0⟩
    >>> state2 = qc.ones(1)   # the state |1⟩
    >>> mixed_state = qc.DensityOperator.from_ensemble(
    ...     [state1, state2],
    ...     ps=[0.5, 0.5]
    ... )  # the statistical ensemble known to be in states |0⟩ and |1⟩
    ...    # each with probability 1/2.
    >>> mixed_state.measure()  # measure the qubit
    (0,)

The :py:meth:`.DensityOperator.measure` method has two arguments:
a list of indices specifying the qubits to be measured,
and a flag specifying whether the measured qubits are to be removed from the state.
If no indices are supplied, every qubit is measured.
E.g., in the following example, the system is known to be in state
:math:`|00⟩` or state :math:`|11⟩` with equal probability. We measure
the first qubit, opting to then remove the collapsed qubit from the state,
and we measure :math:`0`.
Post-measurement, the density operator represents the state :math:`|0⟩`
(for the second qubit) with certainty.

.. code-block:: python

    >>> state1 = qc.bitstring(0, 0)  # the state |00⟩
    >>> state2 = qc.bitstring(1, 1)  # the state |11⟩
    >>> mixed_state = qc.DensityOperator.from_ensemble(
    ...     [state1, state2],
    ...     ps=[0.5, 0.5]
    ... )  # the statistical ensemble known to be in states |00⟩ and |11⟩
    ...    # each with probability 1/2.
    >>> mixed_state.measure(qubit_indices=[0], remove=True)  # measure the state, and remove the
    ...                                                      # qubit from the density operator
    (0,)
    >>> print(mixed_state)
    ... Density operator for 1-qubit state space. Tensor:
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j]]

Generally, the post-measurement state is as follows.
Each of the ensemble of states the density operator represents
collapses to the computational basis state corresponding to the result
of the measurement of the measured qubits.
The mixture probabilities,
representing our state of belief of the current state of the system,
are updated using Bayes rule.
Equivalently, the post-measurement density
operator is :math:`P_m \rho P_m^{\dagger} / p(m)`, where :math:`m` is the
outcome of measurement, :math:`p(m)` is
the probability of that outcome, and :math:`P_m` is the projector onto
the computational basis states corresponding to the measurement outcome,
and :math:`\rho` is the pre-measurement state.

One can take the tensor product of density operators for subsystems,
in the same way as for states and operators, to obtain the density
operator for the larger composite system:

.. code-block:: python

    >>> state1 = qc.DensityOperator.from_ensemble(list_of_states1)
    >>> state2 = qc.DensityOperator.from_ensemble(list_of_states2)
    >>> print(state1.rank)
    2
    >>> print(state2.rank)
    4
    >>> composite_system = state1 * state2
    >>> print(composite_system.rank)
    6


Reduced Density Operators and Purification
==========================================

The reduced density operator of a state can be computed using the
:py:meth:`.State.reduced_density_operator` or
:py:meth:`.DensityOperator.reduced_density_operator` methods,
for pure and mixed states respectively.
This will give the reduced density operator for the subsystem of the given
qubit indices, by tracing out the remaining qubits.
This is equivalent to the density operator obtained by measuring the traced out
qubits and forgetting the result of the measurement.

.. code-block:: python

    >>> state = qc.bell_state(0, 0)
    >>> rho = state.reduced_density_operator(qubit_indices=[0])
    >>> print(rho)
    Density operator for 1-qubit state space. Tensor:
    [[0.5+0.j 0. +0.j]
     [0. +0.j 0.5+0.j]]

A mixed state can be purified with the :py:meth:`.DensityOperator.purify` method.
This will produce a `2d` qubit pure state from a `d` qubit mixed state, with the
measurement probabilities of the first `d` qubits in agreement with the measurement
probabilities of the mixed state.

.. code-block:: python

    >>> state = rho.purify()
    >>> print(state)
    State([[0.70710678+0.j 0.        +0.j]
           [0.        +0.j 0.70710678+0.j]])

Warning: The No-Cloning Theorem
===============================

The `no cloning theorem <https://en.wikipedia.org/wiki/No-cloning_theorem>`_ says that, in general, given a quantum system in a given state,
one cannot clone the state such that another system is in the same state without
modifying the state of the original system.

There are several ways to violate the no-cloning theorem in the QCircuits library,
and users should be aware that none of the following constitute valid steps in a
quantum computation.

Firstly, one could simply create a deepcopy of a state.

Secondly, one could extract the underlying tensor of a state and then create another state from it.

Thirdly, one could apply an operator to a state, such as:

.. code-block:: python

    >>> y = Op(x)

and then use both :math:`x` and :math:`y` in subsequent computation. It is clear that this violates the no-cloning theorem, since the operator may be the identity.

Fourthly, one can take the tensor product of a state with itself, or a tensor power, as in:

.. code-block:: python

    >>> x * x
    >>> x**5

I have decided not to disallow any of these, since each has genuine use-cases. For example
if we know how to prepare a single qubit in a given state, then we know how to prepare
a multi-qubit system with each qubit in that state, and the tensor power is a short-hand way
of achieving this.



Entanglement / Schmidt Number
=============================

A state :math:`|\phi⟩` of a composite quantum system
:math:`A\otimes B` has a Schmidt decomposition:
:math:`|\phi⟩ = \sum_i \lambda_i |i_A⟩|i_B⟩`, where
the *Schmidt coefficients* :math:`\lambda_i` are non-negative and the
:math:`|i_A⟩` and :math:`|i_B⟩` are orthonormal bases for systems
:math:`A` and :math:`B`.
The number of non-zero Schmidt coefficients is a measure of the entanglement
of the two systems, and is called the *Schmidt number*.

In QCircuits, the Schmidt number of a multi-qubit state may be computed with
respect to a partitioning of the qubits into two subsystems with the
:py:meth:`.State.schmidt_number` method, by supplying a list of qubit indices
for one of the subsystems.

E.g., the Bell State :math:`(|00⟩ + |11⟩)/\sqrt{2}`, which is already written
as a Schmidt decomposition, has Schmidt number 2:

.. code-block:: python

    >>> print(qc.bell_state(0, 0).schmidt_number(indices=[0]))
    2

The state :math:`(|00⟩ + |01⟩ + |10⟩ + |11⟩)/2` can be written as a product
:math:`(|0⟩ + |1⟩)/\sqrt{2} \otimes (|0⟩ + |1⟩)/\sqrt{2}`, so it has a
Schmidt number of 1:

.. code-block:: python

    >>> print(qc.positive_superposition(d=2).schmidt_number(indices=[0]))
    1

Any state that is the result of a tensor product of states with respect to
two subsystems with have a Schmidt number 1 with respect to those two subsystems:

.. code-block:: python

    >>> state_A = qc.bitstring(0, 1, 0)
    >>> state_B = qc.bitstring(0, 0, 0, 1)
    >>> state_AB = state_A * state_B
    >>> print(state_AB.schmidt_number(indices=[0, 1, 2]))
    1


Examples
========

For examples of the use of QCircuits to implement quantum algorithms,
see the :ref:`examples page<examples_page>`.
