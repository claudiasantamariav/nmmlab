import numpy as np
from scipy.integrate import solve_ivp


def euler_maruyama(f, z0, t, noise_std=0.0, rng=None):
    """
    Euler-Maruyama simple numerical scheme to solve SDE 
    z(t + dt) = z(t) + f(t, z) * dt + noise
    dW = (random_real + 1j * random_imag) * sqrt(dt)

    """
    if rng is None:
        rng = np.random.default_rng(42)
    dt = t[1] - t[0]
    z0 = np.asarray(z0, dtype=complex) #array of complex numbers
    scalar = z0.ndim == 0 
    z0 = np.atleast_1d(z0) # scalar 1+0j becomes [1+0j]
    N = z0.shape[0]
    z = np.empty((len(t), N), dtype=complex)
    z[0] = z0
    for k in range(len(t) - 1):
        dW = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(dt / 2)
        z[k + 1] = z[k] + np.atleast_1d(f(t[k], z[k])) * dt + noise_std * dW
    # return 1-D array for scalar input (single oscillator), 2-D for networks
    return z[:, 0] if scalar else z


#═══════════════════════════════════════════════════════════════════════════════
# HARMONIC OSCILLATORS
#═══════════════════════════════════════════════════════════════════════════════
""" 
For all models: 
if noise = 0 --> we use solve_ivp 
If noise is not 0 --> use Euler 
"""

#──── SIMPLE HARMONIC OSCILLATOR ──────────────────────────────────────────────────────────────
#If alpha = 0, it's undamped 

def ho(omega, alpha, z0, t, noise_std=0.0, rng=None):
    if alpha == 0:
        # use solve_ivp for accuracy when no noise
        def ode(t_, z):
            return (alpha + 1j * omega) * z[0]
        sol = solve_ivp(ode, (t[0], t[-1]), [z0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]
    else:
        def f(t_, z_):
            return (alpha + 1j * omega) * z_
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)


# ─── DHO WITH SINUSOIDAL FORCING───────────────────────────────────────────────────────────────

def forced_dho(omega, alpha, F0, omega_drive, z0, t, noise_std=0.0, rng=None):
    if noise_std == 0.0:
        def ode(t_, z):
            F = F0 * np.exp(1j * omega_drive * t_)
            return (alpha + 1j * omega) * z[0] + F
        sol = solve_ivp(ode, (t[0], t[-1]), [z0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]
    else:
        def f(t_, z_):
            F = F0 * np.exp(1j * omega_drive * t_)
            return (alpha + 1j * omega) * z_ + F
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)

def resonance_curve(omega0, gamma, F0, omega_range):
    """
    Complex transfer function: X(ω) = F₀ / (ω₀² - ω² + 2iγω)  [from DHO paper section] - this is the steady-state solution's amplitude 
    Returns amplitude |X| and phase angle(X).
    """
    H = F0 / (omega0**2 - omega_range**2 + 2j * gamma * omega_range)
    return np.abs(H)


#──── COUPLED DHO ──────────────────────────────────────────────────────────────

def coupled_dho(alphas, omegas, C, G, z0, t, F, noise_std=0.0, rng=None):
    N = len(z0)
    C_rowsum = C.sum(axis=1)
    if noise_std == 0.0:
        def ode(t_, y):
            z_ = y[:N] + 1j * y[N:]
            coupling = G * (C @ z_ - z_ * C_rowsum)
            dz = (alphas + 1j * omegas) * z_ + coupling + F
            return np.concatenate([dz.real, dz.imag])
        y0 = np.concatenate([z0.real, z0.imag])
        sol = solve_ivp(ode, (t[0], t[-1]), y0, method='RK45',
                        t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[:N].T + 1j * sol.y[N:].T
    else:
        def f(t_, z_):
            coupling = G * (C @ z_ - z_ * C_rowsum)
            return (alphas + 1j * omegas) * z_ + coupling + F
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)