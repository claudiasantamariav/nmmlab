import numpy as np
from scipy.integrate import solve_ivp


#═══════════════════════════════════════════════════════════════════════════════
# NEXT-GENERATION MEAN FIELD (NMM2 — MONTBRIÓ-PAZÓ-ROXIN)
#═══════════════════════════════════════════════════════════════════════════════
# Exact mean-field reduction of a population of all-to-all quadratic
# integrate-and-fire (QIF) neurons with Lorentzian-distributed excitabilities
# (Montbrió, Pazó & Roxin). Replaces the static wave-to-pulse sigmoid of the
# earlier NMMs with a dynamic relation between firing rate r and mean membrane
# potential v. Instantaneous synapses (s = r) are used throughout; a synaptic
# operator L̂ₛ[s] = r (Eq. 7.6) can wrap these to add rise/decay filtering.


#──── MPR SINGLE POPULATION ──────────────────────────────────────────────────────────────
def mpr(eta_bar, delta, J, I, y0, t, noise_std=0.0, rng=None):
    """Montbrió-Pazó-Roxin exact mean field for one QIF population. Rosetta Stone Eq. 7.4-7.5.

    ṙ = Δ/π + 2·r·v
    v̇ = v² − π²·r² + η̄ + J·r + I

    r = population firing rate (stays ≥ 0), v = mean membrane potential. Derived exactly
    from N all-to-all QIF neurons in the large-N limit (Eq. 7.1). η̄ and Δ are the centre and
    half-width-at-half-maximum of the Lorentzian (Cauchy) distribution of neuron
    excitabilities ηⱼ; J is the synaptic self-coupling (instantaneous synapse, s = r).

    Parameters
    ----------
    eta_bar   : float   — centre of the Lorentzian excitability distribution
    delta     : float   — HWHM of the excitability distribution (Δ > 0, heterogeneity)
    J         : float   — synaptic self-coupling strength (J > 0 excitatory)
    I         : float   — common external input / electric-field drive
    y0        : [r0, v0] — initial firing rate and mean membrane potential
    t         : array   — time vector
    noise_std : float   — additive noise on the membrane-potential drive (default 0)
    rng       : np.random.Generator

    Returns sol.y of shape (2, len(t)): rows = [r, v].
    """
    if rng is None:
        rng = np.random.default_rng()

    pi = np.pi

    if noise_std == 0.0:
        def ode(t_, state):
            r, v = state
            dr = delta / pi + 2 * r * v
            dv = v**2 - pi**2 * r**2 + eta_bar + J * r + I
            return [dr, dv]
        sol = solve_ivp(ode, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            r, v = state
            noise = noise_std * rng.standard_normal()
            dr = delta / pi + 2 * r * v
            dv = v**2 - pi**2 * r**2 + eta_bar + J * r + I + noise
            return [dr, dv]
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y


#──── NMM2 E-I PUSH-PULL NODE (PING) ──────────────────────────────────────────────────────────────
def nmm2(eta_x, eta_y, delta_x, delta_y, J_x, J_y, C_xy, C_yx, Fe, I, y0, t, noise_std=0.0, rng=None):
    """Two-population (E-I) next-generation mean-field node. Rosetta Stone Eq. 7.9.

    ṙ_x = Δx/π + 2·r_x·v_x
    v̇_x = v_x² − π²·r_x² + η̄x + J_x·r_x − C_xy·r_y + I + Fe
    ṙ_y = Δy/π + 2·r_y·v_y
    v̇_y = v_y² − π²·r_y² + η̄y − J_y·r_y + C_yx·r_x

    The same E/I push-pull skeleton as WILCO and NMM1, but the static sigmoid is replaced by
    the exact (r, v) dynamics. Instantaneous synapses (s = r). Sign convention (Eq. 7.9):
    J_x > 0 excitatory self-coupling, J_y > 0 inhibitory self-coupling (enters with −),
    C_yx > 0 excitatory E→I (+), C_xy > 0 inhibitory I→E (−). Forcing Fe and common input I
    enter the excitatory population only.

    Parameters
    ----------
    eta_x, eta_y     : float — Lorentzian centres for E and I populations
    delta_x, delta_y : float — Lorentzian HWHMs (Δ > 0)
    J_x, J_y         : float — excitatory / inhibitory self-coupling (≥ 0)
    C_xy             : float — I→E cross-coupling magnitude (enters −)
    C_yx             : float — E→I cross-coupling magnitude (enters +)
    Fe               : float — exogenous drive to the excitatory population
    I                : float — common external input / electric field
    y0               : [r_x, v_x, r_y, v_y] — initial conditions
    t                : array — time vector
    noise_std        : float — additive noise on the excitatory drive (default 0)
    rng              : np.random.Generator

    Returns sol.y of shape (4, len(t)): rows = [r_x, v_x, r_y, v_y].
    """
    if rng is None:
        rng = np.random.default_rng()

    pi = np.pi
    pi2 = pi**2

    if noise_std == 0.0:
        def ode(t_, state):
            rx, vx, ry, vy = state
            drx = delta_x / pi + 2 * rx * vx
            dvx = vx**2 - pi2 * rx**2 + eta_x + J_x * rx - C_xy * ry + I + Fe
            dry = delta_y / pi + 2 * ry * vy
            dvy = vy**2 - pi2 * ry**2 + eta_y - J_y * ry + C_yx * rx
            return [drx, dvx, dry, dvy]
        sol = solve_ivp(ode, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            rx, vx, ry, vy = state
            noise = noise_std * rng.standard_normal()
            drx = delta_x / pi + 2 * rx * vx
            dvx = vx**2 - pi2 * rx**2 + eta_x + J_x * rx - C_xy * ry + I + Fe + noise
            dry = delta_y / pi + 2 * ry * vy
            dvy = vy**2 - pi2 * ry**2 + eta_y - J_y * ry + C_yx * rx
            return [drx, dvx, dry, dvy]
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y


#──── NMM2 NETWORK (E→E coupled E-I motifs) ──────────────────────────────────────────────────────────────
def nmm2_network(eta_x, eta_y, delta_x, delta_y, C_xx, C_xy, C_yx, C_yy, Fe, G, C, y0, t, noise_std=0.0, rng=None):
    """Network of N identical next-generation E-I motifs, coupled E→E. Rosetta Stone Eq. 7.13.

    ṙ_x^i = Δx/π + 2·r_x^i·v_x^i
    v̇_x^i = (v_x^i)² − π²(r_x^i)² + η̄x + C_xx·r_x^i − C_xy·r_y^i + G·Σⱼ Cᵢⱼ·r_x^j + Fe,i
    ṙ_y^i = Δy/π + 2·r_y^i·v_y^i
    v̇_y^i = (v_y^i)² − π²(r_y^i)² + η̄y − C_yy·r_y^i + C_yx·r_x^i

    Each node is an identical E-I motif (instantaneous synapses, s = r); only excitatory
    populations are coupled long-range. All coupling magnitudes are ≥ 0; the signs in the
    voltage equations encode the role: + excitatory (C_xx self, C_yx E→I, long-range C_ij
    E→E), − inhibitory (C_yy self, C_xy I→E).

    Parameters
    ----------
    eta_x, eta_y     : float — Lorentzian centres for E and I
    delta_x, delta_y : float — Lorentzian HWHMs (Δ > 0)
    C_xx             : float — excitatory self-recurrence (+)
    C_xy             : float — I→E within-node coupling (enters −)
    C_yx             : float — E→I within-node coupling (enters +)
    C_yy             : float — inhibitory self-recurrence (enters −)
    Fe               : float or array length N — drive to each excitatory population
    G                : float — global long-range coupling strength
    C                : (N, N) array — connectivity matrix (zeros on diagonal)
    y0               : array length 4N — [rx_0..rx_{N-1}, vx_0.., ry_0.., vy_0..]
    t                : array — time vector
    noise_std        : float — additive noise on the excitatory drive (default 0)
    rng              : np.random.Generator

    Returns sol.y of shape (4N, len(t)).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(C)
    pi = np.pi
    pi2 = pi**2
    Fe_vals = np.broadcast_to(Fe, N)

    if noise_std == 0.0:
        def ode(t_, state):
            rx, vx, ry, vy = state[0:N], state[N:2*N], state[2*N:3*N], state[3*N:4*N]
            coupling = G * (C @ rx)
            drx = delta_x / pi + 2 * rx * vx
            dvx = vx**2 - pi2 * rx**2 + eta_x + C_xx * rx - C_xy * ry + coupling + Fe_vals
            dry = delta_y / pi + 2 * ry * vy
            dvy = vy**2 - pi2 * ry**2 + eta_y - C_yy * ry + C_yx * rx
            return np.concatenate([drx, dvx, dry, dvy])
        sol = solve_ivp(ode, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            rx, vx, ry, vy = state[0:N], state[N:2*N], state[2*N:3*N], state[3*N:4*N]
            coupling = G * (C @ rx)
            noise = noise_std * rng.standard_normal(N)
            drx = delta_x / pi + 2 * rx * vx
            dvx = vx**2 - pi2 * rx**2 + eta_x + C_xx * rx - C_xy * ry + coupling + Fe_vals + noise
            dry = delta_y / pi + 2 * ry * vy
            dvy = vy**2 - pi2 * ry**2 + eta_y - C_yy * ry + C_yx * rx
            return np.concatenate([drx, dvx, dry, dvy])
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
