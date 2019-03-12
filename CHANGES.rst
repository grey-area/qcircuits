v0.3.0, 2019/03/12
==================

- Added methods to permute or swap qubits in a state
- There is now a flag when measuring the state that determines whether or not the measured qubits are removed from the state vector
- Added a method to get the adjoint/inverse of an operator
- Can now measure an arbitrary number of bits at once, in any order
- Measuring a state now modifies it, rather than returning the post-measurement state
- Operators can now be applied to qubits in arbitrary order
- Added Toffoli gate
- Loosened requirements to >python3.4.3 and >= numpy1.11.3
- Added unit tests
- Added documentation

v0.2.0, 2019/03/09 -- API change for Operator construction, and added documentation.

v0.1.0, 2019/03/08 -- Initial release.
