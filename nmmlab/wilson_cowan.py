import numpy as np
from scipy.integrate import solve_ivp


#SIGMOID FUNCTION 
def sigmoid(I, slope, I0):
    return 1/(1 + np.exp(slope*(I0 - I)))


#═══════════════════════════════════════════════════════════════════════════════
# WILSON COWAN
#═══════════════════════════════════════════════════════════════════════════════

#──── WILCO UNFORCED  ──────────────────────────────────────────────────────────────
def wilco_unforced(tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, x0, t, noise_std=0.0):
    if noise_std == 0.0:
        def ode(t_, state):
            x, y = state
            dx = (sigmoid((w_xx * x - w_xy * y + P_x ), slope =1.3, I0=4.0)
                  -x)/tau_x
            dy = (sigmoid((w_yx * x - w_yy * y), slope = 2, I0=3.7) -y)/tau_y
            return [dx, dy]
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

    else:
        def ode_noisy(t_, state):
            x, y = state
            noise = noise_std * np.random.randn()  # additive noise on E
            dx = (sigmoid((w_xx * x - w_xy * y + P_x + noise), slope =1.3, I0=4.0)
                  -x)/tau_x
            dy = (sigmoid((w_yx * x - w_yy * y), slope = 2, I0=3.7) -y)/tau_y
            return [dx, dy]
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

#──── WILCO SINUSOIDALLY FORCED  ──────────────────────────────────────────────────────────────
def wilco_forced(tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, A, f, x0, t, noise_std=0.0):
    if noise_std == 0.0:
        def ode(t_, state):
            x, y = state
            F_e = A * np.sin(2 * np.pi * f * t_ / 1000)
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e, slope=1.3, I0=4.0) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope=2.0, I0=3.7) - y) / tau_y
            return [dx, dy]
        sol = solve_ivp(ode, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y
    else:
        def ode_noisy(t_, state):
            x, y = state
            F_e = A * np.sin(2 * np.pi * f * t_ / 1000)
            noise = noise_std * np.random.randn()
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e + noise, slope=1.3, I0=4.0) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope=2.0, I0=3.7) - y) / tau_y
            return [dx, dy]
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

#──── WILCO NETWORK  ──────────────────────────────────────────────────────────────
def wilco_network(tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, G, C, A, f, x0, t, noise_std=0.0):
    N = len(P_x)
    if noise_std == 0.0:
        def ode(t_, state):
            x = state[:N]
            y = state[N:]
            F_e = np.zeros(N)
            F_e[0] = A * np.sin(2 * np.pi * f * t_ / 1000)
            coupling = G * (C @ x)
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e + coupling, slope=1.3, I0=4.0) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope=2.0, I0=3.7) - y) / tau_y
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
            noise = noise_std * np.random.randn(N)
            dx = (sigmoid(w_xx*x - w_xy*y + P_x + F_e + coupling + noise, slope=1.3, I0=4.0) - x) / tau_x
            dy = (sigmoid(w_yx*x - w_yy*y, slope=2.0, I0=3.7) - y) / tau_y
            return np.concatenate([dx, dy])
        sol = solve_ivp(ode_noisy, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-9)
        return sol.y

