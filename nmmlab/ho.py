import numpy as np
from scipy.integrate import solve_ivp


def euler_maruyama(f, z0, t, noise_std=0.0, rng=None):
    """
    Euler-Maruyama simple numerical scheme to solve SDE 
    z(t + dt) = z(t) + f(t, z) * dt + noise
    dW = (random_real + 1j * random_imag) * sqrt(dt)

    Parameters
    ----------
    f         : callable(t, z) → z  — drift function
    z0        : complex or array of complex — initial condition(s)
    t         : array — time vector
    noise_std : float — noise amplitude (default 0, no noise)
    rng       : np.random.Generator 

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
    """Damped harmonic oscillator. Rosetta Stone Eq. 2.13.

    ż = (α + iω)·z
    α = 0 → undamped, α < 0 → damped, α > 0 → growing.
    Parameters
    ----------
    omega     : float   — natural frequency (rad/s)
    alpha     : float   — damping coefficient (α < 0 damps, α = 0 undamped)
    z0        : complex — initial condition
    t         : array   — time vector
    noise_std : float   — noise amplitude (default 0)
    rng       : np.random.Generator 
    
    returns a complex array of shape (len(t),)
    """
    def f(t_, z_):
        return (alpha + 1j * omega) * z_
    if noise_std == 0.0:
        sol = solve_ivp(lambda t_, z: [f(t_, z[0])], (t[0], t[-1]), [z0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]
    else:
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)
        
#──── KURAMOTO ──────────────────────────────────────────────────────────────
def kuramoto(omegas, G, t, C=None, noise_std=0.0, rng=None):
    """Kuramoto phase oscillator network. Rosetta Stone Appendix D, Eq. D.17.

    dθᵢ/dt = ωᵢ + G · Σⱼ Cᵢⱼ · sin(θⱼ − θᵢ)
    Parameters
    ----------
    omegas    : array of length N — natural frequencies (rad/s)
    G         : float — global coupling strength
    t         : array — time vector
    C         : (N, N) array — connectivity matrix.
                Defaults to all-to-all with zeros on diagonal, normalised by N.
    noise_std : float — phase noise amplitude (default 0)
    rng       : np.random.Generator
    
    returns theta, R 
    """
    
    N = len(omegas)
    if rng is None:
        rng = np.random.default_rng(42)
    if C is None:
        C = np.ones((N, N))
        np.fill_diagonal(C, 0)
        C = C / N  # all-to-all, row-normalised
    dt = t[1] - t[0]
    theta = np.empty((len(t), N))
    theta[0] = rng.uniform(0, 2 * np.pi, N)
    for k in range(len(t) - 1):
        diffs = theta[k][np.newaxis, :] - theta[k][:, np.newaxis]  # [i,j] = θ_j - θ_i
        coupling = G * np.sum(C * np.sin(diffs), axis=1)
        noise = noise_std * rng.standard_normal(N) * np.sqrt(dt)
        theta[k + 1] = theta[k] + (omegas + coupling) * dt + noise
    R = np.abs(np.mean(np.exp(1j * theta), axis=1))
    return theta, R


# ─── DHO WITH SINUSOIDAL FORCING───────────────────────────────────────────────────────────────

def forced_dho(omega, alpha, F0, omega_drive, z0, t, noise_std=0.0, rng=None):
    """Sinusoidally forced damped harmonic oscillator. Rosetta Stone Eq. 2.19.

    ż = (α + iω)·z + F₀·e^{iΩt}
     Parameters
    ----------
    omega       : float   — natural frequency (rad/s)
    alpha       : float   — damping coefficient (α < 0 for stable oscillation)
    F0          : float   — forcing amplitude
    omega_drive : float   — driving frequency Ω (rad/s)
    z0          : complex — initial condition
    t           : array   — time vector
    noise_std   : float   — noise amplitude (default 0)
    rng         : np.random.Generator
    
    returns a complex array of shape (len(t),)
    
    """
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
    Parameters
    ----------
    omega0      : float — natural frequency ω₀ (rad/s)
    gamma       : float — damping coefficient γ (γ > 0)
    F0          : float — forcing amplitude
    omega_range : array — driving frequencies Ω to sweep over
    
    Returns amplitude |X| and phase angle(X).
    """
    H = F0 / (omega0**2 - omega_range**2 + 2j * gamma * omega_range)
    return np.abs(H), np.angle(H, deg=True)


#──── COUPLED DHO ──────────────────────────────────────────────────────────────

def coupled_dho(alphas, omegas, C, G, z0, t, F, noise_std=0.0, rng=None):
    """Network of N diffusively coupled DHOs. Rosetta Stone Eq. 2.23.

    żᵢ = (αᵢ + iωᵢ)·zᵢ + G·Σⱼ Cᵢⱼ·(zⱼ − zᵢ) + Fᵢ
    Parameters
    ----------
    alphas    : array of length N — damping per node
    omegas    : array of length N — natural frequency per node (rad/s)
    C         : (N, N) array     — connectivity matrix (zeros on diagonal)
    G         : float            — global coupling strength
    z0        : complex array of length N — initial conditions
    t         : array            — time vector
    F         : complex array of length N — constant external forcing per node
    noise_std : float            — noise amplitude (default 0)
    rng       : np.random.Generator
    
    return (len(t), N) complex array
    """
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




