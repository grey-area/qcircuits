import qcircuits as qc
import numpy as np
from scipy.stats import unitary_group


# Phase estimation
# We are given a black-box unitary operator, and one of its eigenstates.
# The task is to estimate the phase of the corresponding eigenvalue.
# This is done making use of the efficient inverse quantum Fourier transform.


# Prepares a state that when the inverse Fourier transform is applied,
# unpacks the binary fractional expansion of the phase into the
# first t-qubit register.
def stage_1(state, U, t, d):
    state = qc.Hadamard(d=t)(state, qubit_indices=range(t))

    # For each qubit in reverse order, apply the Hadamard gate,
    # then apply U^(2^i) to the d-qubit register
    # conditional on the state of the t-i qubit in the
    # t-qubit register.
    for idx, t_i in enumerate(range(t-1, -1, -1)):
        U_2_idx = qc.Identity(d)
        for app_i in range(2**idx):
            U_2_idx = U(U_2_idx)
        C_U = qc.ControlledU(U_2_idx)
        state = C_U(
            state,
            qubit_indices=[t_i] + list(range(t, t+d, 1))
        )
        
    return state


# The t-qubit quantum Fourier transform
def QFT(t):
    Op = qc.Identity(t)
    H = qc.Hadamard()

    # The R_k gate applies a 2pi/2^k phase is the qubit is set
    C_Rs = {}
    for k in range(2, t+1, 1):
        R_k = np.exp(np.pi * 1j / 2**k) * qc.RotationZ(2*np.pi / 2**k)
        C_Rs[k] = qc.ControlledU(R_k)

    # For each qubit in order, apply the Hadamard gate, and then
    # apply the R_2, R_3, ... conditional on the remainder of the qubits
    for t_i in range(t):
        Op = H(Op, qubit_indices=[t_i])
        for k in range(2, t+1 - t_i, 1):
            Op = C_Rs[k](Op, qubit_indices=[t_i + k - 1, t_i])

    # We have the QFT, but with the qubits in reverse order
    # Swap them back
    Swap = qc.Swap()
    for i, j in zip(range(t), range(t-1, -1, -1)):
        if i >= j:
            break
        Op = Swap(Op, qubit_indices=[i, j])

    return Op


# The t-qubit inverse quantum Fourier transform
def inv_QFT(t):
    return QFT(t).adj


# Do phase estimation for a random d-qubit operator,
# recording the result in a t-qubit register.
def phase_estimation(d=2, t=8):
    # a d-qubit gate
    U = unitary_group.rvs(2**d)
    eigvals, eigvecs = np.linalg.eig(U)    
    U = qc.Operator.from_matrix(U)
    
    # an eigenvector u and the phase of its eigenvalue, phi
    phi = np.real(np.log(eigvals[0]) / (2j*np.pi))
    if phi < 0:
        phi += 1
    u = eigvecs[:, 0]
    u = qc.State.from_column_vector(u)
    
    # add the t-qubit register
    state = qc.zeros(t) * u

    state = stage_1(state, U, t, d)
    state = inv_QFT(t)(state, qubit_indices=range(t))
    measurement = state.measure(qubit_indices=range(t))
    phi_estimate = sum(measurement[i] * 2**(-i-1) for i in range(t))

    return phi, phi_estimate

    
if __name__ == '__main__':
    phi, phi_estimate = phase_estimation(d=2, t=8)

    print('True phase: {}'.format(phi))
    print('Estimated phase: {}'.format(phi))
