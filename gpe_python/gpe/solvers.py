import numpy as np
from gpe.common import V_full



# Time Splitting Spectral Method 

# Kinetic Part
def TSSM_kinetic_step(phi, dx, dt):

    N_ext = phi.size
    k = 2* np.pi * np.fft.fftfreq(N_ext, d=dx)
    phi_k= np.fft.fft(phi)
    phi_k *= np.exp(-0.5 * dt * (k**2))   # e^{-dt k^2 / 2}
    phi_star = np.fft.ifft(phi_k)
    return phi_star

# TSSM Full Step
def TSSM_step(phi, x, dx, dt, g):
    V1 = V_full(phi, x, g)
    phi_half = np.exp(-0.5 * dt * V1) * phi

    phi_mid = TSSM_kinetic_step(phi_half, dx, dt)

    V2 = V_full(phi_mid, x, g)
    phi_new = np.exp(-0.5 * dt * V2) * phi_mid

    return phi_new



# Forward Euler Method

# Kinetic Part
def laplacian(phi, dx):
    lap = np.zeros_like(phi)
    lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / (dx * dx)
    return lap

# FE Full Step
def FE_step(phi, x, dx, dt, g):
    V = V_full(phi, x, g)
    H_phi = -0.5 * laplacian(phi, dx) + V * phi
    phi_new = phi - dt * H_phi
    return phi_new



# Crank-Nicolson Method

def CN_step(phi, x, dx, dt, g):
    import scipy.linalg as sci

    V = V_full(phi, x, g)
    alpha = dt / (4 * dx * dx)   # off-diagonal magnitude

    # main-diagonal addition from kinetic term is 2*alpha
    first_diagonal_CN = (2*alpha) + (dt/2)*V

    # banded matrix A for (I + dt/2 H): shape (3, N_ext)
    N_ext = phi.size
    A = np.zeros((3, N_ext), dtype=complex)
    A[0, 1:]  = -alpha              # super-diagonal
    A[1, :]   = 1.0 + first_diagonal_CN
    A[2, :-1] = -alpha              # sub-diagonal

    # RHS: (I - dt/2 H) psi^n
    rhs = (1.0 - first_diagonal_CN) * phi
    rhs[:-1] += alpha * phi[1:]
    rhs[1:]  += alpha * phi[:-1]

    phi_new = sci.solve_banded((1, 1), A, rhs)

    return phi_new
