import numpy as np

# Evaluating dt_list expressions

def evaluate_dt_list(dt_list, dr_like):
    out = []
    for d in dt_list:
        if isinstance(d, (int, float)):
            out.append(float(d))
        else:
            out.append(float(eval(d, {"__builtins__": {}}, {"dr": dr_like})))
    return out

#Initial mathematical wavefunction not real one since u=r*R

def initial_phi(r):
    return r * np.exp(-r*r)

# Normalization on Full Grid [-R,R]

def normalize(phi, dr):
    N_half= phi.size // 2
    phi_half = phi[int(N_half):]
    nrm = np.sqrt(4*np.pi * np.sum(np.abs(phi_half)**2) * dr)
    return phi / (nrm + 1e-300)

# Expectation Value of Energy

def expectation_value_energy(phi,x, dr, g):

    dphi = np.empty_like(phi)
    #Kinetic Part
    dphi[0]  = 0.0
    dphi[-1] = 0.0
    dphi[1:-1] = (phi[2:] - phi[:-2])/(2*dr)
    dphi_half = dphi[dphi.size//2:]
    kin = 0.5 * 4*np.pi * np.sum(np.abs(dphi_half)**2) * dr
    #Potential & Non-linear Part
    half = phi.size // 2
    r = x[half:]          # if x is full-line grid
    phi_half = phi[half:] # same length
    pot = 4*np.pi * np.sum(0.5 * r**2 * np.abs(phi_half)**2) * dr
    non = 0.5 * 4*np.pi * g * np.sum((np.abs(phi_half)**4)/(r**2 + 1e-300)) * dr

    return kin + pot + non

# Full Grid from [-R,R]

def full_grid(N, R):
    dr = R / N
    r = np.linspace(-R + dr/2, R - dr/2, 2*N)
    return r, dr

# Potential + Non-linear terms there is also  an adjusment to avoid division by zero. When r<1e-6 the nearst non-zero value is used. V_eff(x) = 1/2 r^2 + g u^2 / r^2

def V_full(phi, x, g):
    r = np.abs(x)
    r2 = r**2
    dens = np.abs(phi)**2

    V = 0.5 * r2
    V_eff = V.copy()

    mask = r2 > 1e-12
    if np.any(mask):
        V_eff[mask] += g * dens[mask] / r2[mask]

    # regularize all near-zero points
    if not np.all(mask) and np.any(mask):
        idx0 = np.where(~mask)[0]
        j_ref = np.where(mask)[0][0]
        V_eff[idx0] = V[idx0] + g * dens[j_ref] / r2[j_ref]

    return V_eff

