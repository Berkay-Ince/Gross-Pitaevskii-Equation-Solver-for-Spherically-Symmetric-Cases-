import time
import gpe.common
import gpe.solvers

# Based on Imaginary Time Propagation Method for The Radially Symmetric Gross–Pitaevskii Equation

def run_method(method, g, dt, N, R, tol, max_iter, renorm_every, report_every, E_aim):

    # Domnain 

    r,dr = gpe.common.full_grid(N, R)

    # Initial Wavefunction

    phi = gpe.common.initial_phi(r)

    # Normalization & Previous Energy

    phi = gpe.common.normalize(phi, dr)
    E_prev = gpe.common.expectation_value_energy(phi, r, dr, g)

    # choose the step function

    if method == "TSSM":
        step_fn = lambda p: gpe.solvers.TSSM_step(p, r, dr, dt, g)
    elif method == "CN":
        step_fn = lambda p: gpe.solvers.CN_step(p, r, dr, dt, g)
    elif method == "FE":
        step_fn = lambda p: gpe.solvers.FE_step(p, r, dr, dt, g)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    steps = 0
    t0 = time.time()
    while steps < max_iter:
        phi = step_fn(phi)
        steps += 1

        if steps % renorm_every == 0:
            phi = gpe.common.normalize(phi, dr)
        
        if steps % report_every == 0:
            E_curr = gpe.common.expectation_value_energy(phi,r,dr,g)
            rel = abs(E_curr - E_prev)/(abs(E_curr) + 1e-12)
            if rel < tol:
                break
            if E_aim is not None:
                if E_curr < E_aim:
                    break
            E_prev = E_curr

    wall_time = time.time() - t0

    E_final = gpe.common.expectation_value_energy(phi, r, dr, g)
    return {
        "method": method,
        "g": g,
        "dt": dt,
        "N": N,
        "R": R,
        "steps": steps,
        "energy": E_final,
        "wall_time_s": wall_time,
        "r": r,
        "phi": phi,
    }

