# src/simulation.py
import numpy as np
import pandas as pd

def simulate_sveir(params, T=100, dt=0.1, seed=None):
    """Simulate SVEIR model using Euler–Maruyama."""
    
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt) + 1
    t = np.linspace(0, T, n_steps)

    # parameters
    beta = params.get('beta', 0.3)
    nu = params.get('nu', 0.05)
    kappa = params.get('kappa', 0.2)
    gamma = params.get('gamma', 0.1)
    sigma = params.get('sigma', 0.0)
    N = params.get('N', 1000)

    # state arrays
    S = np.zeros(n_steps)
    V = np.zeros(n_steps)
    E = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)

    # initial conditions
    S[0] = params.get('S0', N - 10)
    V[0] = params.get('V0', 0)
    E[0] = params.get('E0', 5)
    I[0] = params.get('I0', 5)
    R[0] = params.get('R0', 0)

    for k in range(n_steps - 1):
        St, Vt, Et, It, Rt = S[k], V[k], E[k], I[k], R[k]

        # deterministic drifts
        dS_det = -beta * St * It / N - nu * St
        dV_det = nu * St
        dE_det = beta * St * It / N - kappa * Et
        dI_det = kappa * Et - gamma * It
        dR_det = gamma * It

        # stochastic noise
        dW = np.random.normal(scale=np.sqrt(dt))

        dS = dS_det * dt + sigma * St * dW
        dV = dV_det * dt + sigma * Vt * dW
        dE = dE_det * dt + sigma * Et * dW
        dI = dI_det * dt + sigma * It * dW
        dR = dR_det * dt + sigma * Rt * dW

        S[k+1] = max(St + dS, 0)
        V[k+1] = max(Vt + dV, 0)
        E[k+1] = max(Et + dE, 0)
        I[k+1] = max(It + dI, 0)
        R[k+1] = max(Rt + dR, 0)

        # optional normalization
        total = S[k+1] + V[k+1] + E[k+1] + I[k+1] + R[k+1]
        if total > 0:
            S[k+1] *= N / total
            V[k+1] *= N / total
            E[k+1] *= N / total
            I[k+1] *= N / total
            R[k+1] *= N / total

    # return DataFrame
    df = pd.DataFrame({
        "t": t,
        "S": S,
        "V": V,
        "E": E,
        "I": I,
        "R": R
    })

    return df
