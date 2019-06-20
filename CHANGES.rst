v0.5.0, 2019/06/20
------------------

* Added density operators, along with operator application, qubit permutation, and measurement functionality.

v0.4.0, 2019/04/22
------------------

* Supports Python 3.4.3
* Added phase gate, pi/8 gate.
* Added rotation gates.
* Can now add, subtract, negate, scalar-multiply states (and operators).
* State re-normalization is now done after operator application and measurement.
* New __str__ and __repr__ methods of operators and states.
* Can now get Kronecker product column vector/matrix representations of states and operators, and construct them from those representations.
* Can now compute the Schmidt number of a composite state.
* Can now permute the incoming and outgoing qubits of an operator in the same way as can be done to states.
* Can now apply d-qubit operators to n-qubit operators where n>d by supplying the indices of the higher-rank operator that the lower-rank operator will apply to. Unified implementation with implementation of applying operator to larger state, simplifying the __call__ method of Operator.

v0.3.0, 2019/03/12
------------------

* Added methods to permute or swap qubits in a state
* There is now a flag when measuring the state that determines whether or not the measured qubits are removed from the state vector
* Added a method to get the adjoint/inverse of an operator
* Can now measure an arbitrary number of bits at once, in any order
* Measuring a state now modifies it, rather than returning the post-measurement state
* Operators can now be applied to qubits in arbitrary order
* Added Toffoli gate
* Loosened requirements to >python3.4.3 and >= numpy1.11.3
* Added unit tests
* Added documentation

v0.2.0, 2019/03/09
------------------

* API change for Operator construction, and added documentation.

v0.1.0, 2019/03/08
------------------

* Initial release.
