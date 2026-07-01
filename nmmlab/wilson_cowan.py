import numpy as np
from scipy.integrate import solve_ivp


#SIGMOID FUNCTION 
def sigmoid(I, slope, I0):
    """Sigmoid transfer function. Rosetta Stone Eq. 4.6.

    σ(I) = 1 / (1 + exp(slope · (I0 − I)))
    Maps any input current to a firing rate in (0, 1).

    Parameters
    ----------
    I     : float or array — input current
    slope : float          — steepness
    I0    : float          — threshold (inflection point)

    Returns float or array in (0, 1)
    """
    return 1/(1 + np.exp(slope*(I0 - I)))


#═══════════════════════════════════════════════════════════════════════════════
# WILSON COWAN
#═══════════════════════════════════════════════════════════════════════════════

#──── WILCO UNFORCED  ──────────────────────────────────────────────────────────────
def wilco_unforced(tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, x0, t, slope_x=1.3, I0_x=4.0, slope_y=2.0, I0_y=3.7,
                   noise_std=0.0, rng=None):
    """Unforced Wilson–Cowan node. Rosetta Stone Eq. 5.2.
    τ_x ẋ + x = σ_x(w_xx·x − w_xy·y + P_x)
    τ_y ẏ + y = σ_y(w_yx·x − w_yy·y)

    x = excitatory firing rate, y = inhibitory firing rate.
    Oscillation emerges from E→I→E push-pull loop above the Hopf bifurcation.

    Parameters
    ----------
    tau_x, tau_y : float        — time constants for E and I (ms)
    w_xx         : float        — E→E weight
    w_xy         : float        — I→E weight
    w_yx         : float        — E→I weight
    w_yy         : float        — I→I weight
    P_x          : float        — external drive to E
    x0           : [float, float] — initial conditions [x, y]
    t            : array        — time vector (ms)
    slope_x, I0_x: float        — sigmoid params for E (default 1.3, 4.0)
    slope_y, I0_y: float        — sigmoid params for I (default 2.0, 3.7)
    noise_std    : float        — additive noise on E (default 0)
    rng          : np.random.Generator 
    
    Returns sol.y of shape (2, len(t)).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if noise_std == 0.0:
        def ode(t_, state):
            x, y = state
            dx = (sigmoid(w_xx * x - w_xy * y + P_x, slope_x, I0_x)-x)/tau_x
            dy = (sigmoid(w_yx * x - w_yy * y, slope_y, I0_y) -y)/tau_y
            return [dx, dy]
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

    else:
        def ode_noisy(t_, state):
            x, y = state
            noise = noise_std * rng.standard_normal()  # additive noise on E
            
            dx = (sigmoid(w_xx * x - w_xy * y + P_x + noise, slope_x, I0_x) - x) / tau_x
            dy = (sigmoid(w_yx * x - w_yy * y,                slope_y, I0_y) - y) / tau_y
            return [dx, dy]
        
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

#──── WILCO SINUSOIDALLY FORCED  ──────────────────────────────────────────────────────────────
def wilco_forced(tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, A, f, x0, t, slope_x=1.3, I0_x=4.0, slope_y=2.0, I0_y=3.7,
                 noise_std=0.0, rng=None):
    """ Sinusoidally forced Wilson–Cowan node. Rosetta Stone Eq. 5.3.
    F_e(t) = A·sin(2π·f·t/1000) added to E input. 

    Parameters
    ----------
    A    : float — forcing amplitude
    f    : float — forcing frequency (Hz)
    
    Returns sol.y of shape (2, len(t)).
    """
    
    if rng is None:
        rng = np.random.default_rng()
        
    if noise_std == 0.0:
        def ode(t_, state):
            x, y = state
            F_e = A * np.sin(2 * np.pi * f * t_ / 1000)
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e, slope_x, I0_x) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope_y, I0_y) - y) / tau_y
            return [dx, dy]
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            x, y = state
            F_e = A * np.sin(2 * np.pi * f * t_ / 1000)
            noise = noise_std * rng.standard_normal()
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e + noise, slope_x, I0_x) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope_y, I0_y) - y) / tau_y
            return [dx, dy]
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

#──── WILCO NETWORK  ──────────────────────────────────────────────────────────────
def wilco_network(tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, G, C, A, f, x0, t, slope_x=1.3, I0_x=4.0, slope_y=2.0, I0_y=3.7,
                  noise_std=0.0, rng=None):
    
    """Network of N coupled Wilson–Cowan nodes. Rosetta Stone Eq. 5.5.
    τ_x ẋᵢ + xᵢ = σ_x(w_xx·xᵢ − w_xy·yᵢ + P_x,i + F̂_e,i + G·Σⱼ Cᵢⱼ·xⱼ)
    τ_y ẏᵢ + yᵢ = σ_y(w_yx·xᵢ − w_yy·yᵢ)

    Forcing applied to node 0 only. 
    Parameters
    ----------
    P_x  : array length N   — external drive per node
    G    : float            — coupling strength
    C    : (N, N) array     — connectivity matrix (zeros on diagonal)
    A    : float            — forcing amplitude (node 0 only)
    f    : float            — forcing frequency (Hz)
    x0   : array length 2N  — initial conditions [x_0..x_N, y_0..y_N]

    Returns sol.y of shape (2N, len(t)).
    """

    if rng is None:
        rng = np.random.default_rng()
        
    N = len(P_x)
    if noise_std == 0.0:
        def ode(t_, state):
            x = state[:N]
            y = state[N:]
            F_e = np.zeros(N)
            F_e[0] = A * np.sin(2 * np.pi * f * t_ / 1000)
            coupling = G * (C @ x)
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e + coupling, slope_x, I0_x) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope_y, I0_y) - y) / tau_y
            return np.concatenate([dx, dy])
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            x = state[:N]
            y = state[N:]
            F_e = np.zeros(N)
            F_e[0] = A * np.sin(2 * np.pi * f * t_ / 1000)
            coupling = G * (C @ x)
            noise = noise_std * rng.standard_normal(N)
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e + coupling + noise, slope_x, I0_x) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope_y, I0_y) - y) / tau_y
            return np.concatenate([dx, dy])
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

