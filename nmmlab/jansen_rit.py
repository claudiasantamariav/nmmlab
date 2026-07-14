import numpy as np
from scipy.integrate import solve_ivp


#SIGMOID FUNCTION
def sigmoid(v, a):
    """Static sigmoid transfer function. Rosetta Stone §6.

    σ(v) = tanh(a·v)
    Maps membrane perturbation to a firing-rate-like output in (-1, 1).

    Parameters
    ----------
    v : float or array — membrane perturbation
    a : float or array — slope (excitability)

    Returns float or array in (-1, 1)
    """
    return np.tanh(a * v)


#═══════════════════════════════════════════════════════════════════════════════
# JANSEN-RIT (NMM1 — SECOND-ORDER SYNAPSES)
#═══════════════════════════════════════════════════════════════════════════════

def k_crit(tau_x, tau_y):
    """Barkhausen loop-gain threshold for sustained NMM1 oscillation. Rosetta Stone Eq. 6.4.

    K_crit = 2 + τx/τy + τy/τx   (≥ 4, minimal when τx = τy)

    Parameters
    ----------
    tau_x, tau_y : float — excitatory / inhibitory synaptic time constants

    Returns float
    """
    return 2 + tau_x / tau_y + tau_y / tau_x


#──── NMM1 SINGLE NODE ──────────────────────────────────────────────────────────────
def nmm1(tau_x, tau_y, gx, gy, wxy, wyx, a, Fe, x0, t, noise_std=0.0, rng=None):
    """Jansen-Rit-style push-pull node with second-order synapses. Rosetta Stone Eq. 6.1.

    τx²ẍ + 2τx·ẋ + x = γx·σ(−wxy·y + Fe)
    τy²ÿ + 2τy·ẏ + y = γy·σ(wyx·x)

    x, y = excitatory / inhibitory postsynaptic potentials (not firing rates).
    No self-coupling needed: the synapses' own phase lag sustains the rhythm once the
    loop gain K = γx·γy·wxy·wyx·a² exceeds K_crit = 2 + τx/τy + τy/τx (Barkhausen criterion).

    Parameters
    ----------
    tau_x, tau_y : float          — excitatory / inhibitory synaptic time constants
    gx, gy       : float          — excitatory / inhibitory synaptic gains
    wxy          : float          — inhibitory→excitatory weight ("pull")
    wyx          : float          — excitatory→inhibitory weight ("push")
    a            : float          — sigmoid slope (excitability)
    Fe           : float          — exogenous drive
    x0           : [x, xdot, y, ydot] — initial conditions
    t            : array          — time vector
    noise_std    : float          — additive noise on the excitatory drive (default 0)
    rng          : np.random.Generator

    Returns sol.y of shape (4, len(t)): rows = [x, xdot, y, ydot].
    """
    if rng is None:
        rng = np.random.default_rng()

    if noise_std == 0.0:
        def ode(t_, state):
            x, vx, y, vy = state
            dvx = (gx * sigmoid(-wxy * y + Fe, a) - x - 2 * tau_x * vx) / tau_x**2
            dvy = (gy * sigmoid(wyx * x, a) - y - 2 * tau_y * vy) / tau_y**2
            return [vx, dvx, vy, dvy]
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            x, vx, y, vy = state
            noise = noise_std * rng.standard_normal()
            dvx = (gx * sigmoid(-wxy * y + Fe + noise, a) - x - 2 * tau_x * vx) / tau_x**2
            dvy = (gy * sigmoid(wyx * x, a) - y - 2 * tau_y * vy) / tau_y**2
            return [vx, dvx, vy, dvy]
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y


#──── NMM1 NETWORK ──────────────────────────────────────────────────────────────
def nmm1_network(tau_x, tau_y, gx, gy, wxy, wyx, a, Fe, G, C, x0, t, noise_std=0.0, rng=None):
    """Network of N coupled NMM1 nodes. Rosetta Stone Eq. 6.6.

    τx²ẍᵢ + 2τx·ẋᵢ + xᵢ = γx·σ(−wxy·yᵢ + Fe,i + G·Σⱼ Cᵢⱼ·xⱼ)
    τy²ÿᵢ + 2τy·ẏᵢ + yᵢ = γy·σ(wyx·xᵢ)

    Long-range coupling enters the excitatory sigmoid only.

    Parameters
    ----------
    tau_x, tau_y : float           — excitatory / inhibitory synaptic time constants
    gx, gy       : float           — excitatory / inhibitory synaptic gains
    wxy          : float           — inhibitory→excitatory weight ("pull")
    wyx          : float           — excitatory→inhibitory weight ("push")
    a            : float or array length N — sigmoid slope per node (excitability); heterogeneity
                   gives the coupling something to synchronize
    Fe           : float or array length N — exogenous drive per node
    G            : float           — coupling strength
    C            : (N, N) array    — connectivity matrix (zeros on diagonal)
    x0           : array length 4N — initial conditions [x_0..x_N, xdot_0..xdot_N, y_0..y_N, ydot_0..ydot_N]
    t            : array           — time vector
    noise_std    : float           — additive noise on the excitatory drive (default 0)
    rng          : np.random.Generator

    Returns sol.y of shape (4N, len(t)).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(C)
    a_vals = np.broadcast_to(a, N)
    Fe_vals = np.broadcast_to(Fe, N)

    if noise_std == 0.0:
        def ode(t_, state):
            x, vx, y, vy = state[0:N], state[N:2*N], state[2*N:3*N], state[3*N:4*N]
            coupling = G * (C @ x)
            dvx = (gx * sigmoid(-wxy * y + Fe_vals + coupling, a_vals) - x - 2 * tau_x * vx) / tau_x**2
            dvy = (gy * sigmoid(wyx * x, a_vals) - y - 2 * tau_y * vy) / tau_y**2
            return np.concatenate([vx, dvx, vy, dvy])
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            x, vx, y, vy = state[0:N], state[N:2*N], state[2*N:3*N], state[3*N:4*N]
            coupling = G * (C @ x)
            noise = noise_std * rng.standard_normal(N)
            dvx = (gx * sigmoid(-wxy * y + Fe_vals + coupling + noise, a_vals) - x - 2 * tau_x * vx) / tau_x**2
            dvy = (gy * sigmoid(wyx * x, a_vals) - y - 2 * tau_y * vy) / tau_y**2
            return np.concatenate([vx, dvx, vy, dvy])
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
