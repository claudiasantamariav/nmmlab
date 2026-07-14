import numpy as np
from scipy.integrate import solve_ivp
from nmmlab.ho import euler_maruyama


#═══════════════════════════════════════════════════════════════════════════════
# STUART-LANDAU
#═══════════════════════════════════════════════════════════════════════════════

#──── SIMPLE STUART-LANDAU OSCILLATOR ──────────────────────────────────────────────────────────────
def sl(alpha, omega, gamma, beta, z0, t, noise_std=0.0, rng=None):
    """Stuart-Landau oscillator — Hopf normal form. Rosetta Stone Eq. 3.6.

    ż = (α + iω)·z − (γ + iβ)·|z|²·z
    α < 0 → decay to rest, α = 0 → Hopf bifurcation, α > 0 → limit cycle at r* = √(α/γ).

    Parameters
    ----------
    alpha     : float   — linear growth/decay rate (Hopf bifurcation parameter)
    omega     : float   — intrinsic angular frequency (rad/s)
    gamma     : float   — nonlinear amplitude saturation (γ > 0)
    beta      : float   — amplitude-phase coupling ("shear")
    z0        : complex — initial condition
    t         : array   — time vector
    noise_std : float   — noise amplitude (default 0)
    rng       : np.random.Generator

    returns a complex array of shape (len(t),)
    """
    def f(t_, z_):
        return (alpha + 1j * omega) * z_ - (gamma + 1j * beta) * np.abs(z_)**2 * z_
    if noise_std == 0.0:
        sol = solve_ivp(lambda t_, z: [f(t_, z[0])], (t[0], t[-1]), [z0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]
    else:
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)


#──── SL WITH SINUSOIDAL FORCING ──────────────────────────────────────────────────────────────
def sl_forced(alpha, omega, gamma, beta, F0, omega_drive, z0, t, noise_std=0.0, rng=None):
    """Sinusoidally forced Stuart-Landau oscillator. Rosetta Stone Eq. 3.7.

    ż = (α + iω)·z − (γ + iβ)·|z|²·z + F₀·e^{iΩt}

    Parameters
    ----------
    alpha       : float   — linear growth/decay rate (Hopf bifurcation parameter)
    omega       : float   — intrinsic angular frequency (rad/s)
    gamma       : float   — nonlinear amplitude saturation (γ > 0)
    beta        : float   — amplitude-phase coupling ("shear")
    F0          : float   — forcing amplitude
    omega_drive : float   — driving frequency Ω (rad/s)
    z0          : complex — initial condition
    t           : array   — time vector
    noise_std   : float   — noise amplitude (default 0)
    rng         : np.random.Generator

    returns a complex array of shape (len(t),)
    """
    def f(t_, z_):
        F = F0 * np.exp(1j * omega_drive * t_)
        return (alpha + 1j * omega) * z_ - (gamma + 1j * beta) * np.abs(z_)**2 * z_ + F
    if noise_std == 0.0:
        sol = solve_ivp(lambda t_, z: [f(t_, z[0])], (t[0], t[-1]), [z0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]
    else:
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)


def bifurcation_curve(alpha_range, gamma):
    """Analytic long-run amplitude r*(α) for the SL Hopf bifurcation. Rosetta Stone Eq. 3.6.

    r* = 0 for α ≤ 0 (stable fixed point), r* = √(α/γ) for α > 0 (limit cycle).

    Parameters
    ----------
    alpha_range : array — sweep of α values
    gamma       : float — nonlinear amplitude saturation (γ > 0)

    Returns (r_stable, r_unstable, r_limit_cycle) — each same shape as alpha_range,
    NaN outside the regime it describes.
    """
    r_stable   = np.where(alpha_range < 0, 0.0, np.nan)
    r_unstable = np.where(alpha_range >= 0, 0.0, np.nan)
    r_lc       = np.where(alpha_range > 0, np.sqrt(np.maximum(alpha_range, 0) / gamma), np.nan)
    return r_stable, r_unstable, r_lc


#──── COUPLED STUART-LANDAU NETWORK ──────────────────────────────────────────────────────────────
def sl_network(alphas, omegas, gammas, betas, C, G, z0, t, F=0.0, noise_std=0.0, rng=None):
    """Network of N diffusively coupled Stuart-Landau oscillators. Rosetta Stone Eq. 3.8-3.9.

    żᵢ = (αᵢ + iωᵢ)·zᵢ − (γᵢ + iβᵢ)·|zᵢ|²·zᵢ + G·Σⱼ Cᵢⱼ·(zⱼ − zᵢ) + Fᵢ

    Parameters
    ----------
    alphas    : array of length N — linear growth/decay rate per node
    omegas    : array of length N — intrinsic angular frequency per node (rad/s)
    gammas    : array of length N — nonlinear amplitude saturation per node (> 0)
    betas     : array of length N — amplitude-phase coupling per node
    C         : (N, N) array     — connectivity matrix (zeros on diagonal)
    G         : float            — global coupling strength
    z0        : complex array of length N — initial conditions
    t         : array            — time vector
    F         : complex array of length N — constant external forcing per node (default 0)
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
            dz = (alphas + 1j * omegas) * z_ - (gammas + 1j * betas) * np.abs(z_)**2 * z_ + coupling + F
            return np.concatenate([dz.real, dz.imag])
        y0 = np.concatenate([z0.real, z0.imag])
        sol = solve_ivp(ode, (t[0], t[-1]), y0, method='RK45',
                        t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[:N].T + 1j * sol.y[N:].T
    else:
        def f(t_, z_):
            coupling = G * (C @ z_ - z_ * C_rowsum)
            return (alphas + 1j * omegas) * z_ - (gammas + 1j * betas) * np.abs(z_)**2 * z_ + coupling + F
        return euler_maruyama(f, z0, t, noise_std=noise_std, rng=rng)
