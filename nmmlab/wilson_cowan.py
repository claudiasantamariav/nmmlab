import numpy as np
from scipy.integrate import solve_ivp


#SIGMOID FUNCTION 
def sigmoid(I, slope, I0):
    return 1/(1 + np.exp(slope*(I0 - I)))


#═══════════════════════════════════════════════════════════════════════════════
# WILSON COWAN
#═══════════════════════════════════════════════════════════════════════════════

#──── WILCO UNFORCED  ──────────────────────────────────────────────────────────────
def wilco_unforced(t, state, tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, noise_std=0.0):
    x, y = state
    noise = noise_std * np.random.randn()  # additive noise on E
    dx = (sigmoid((w_xx * x - w_xy * y + P_x + noise), slope =1.3, I0=4.0)
          -x)/tau_x
    dy = (sigmoid((w_yx * x - w_yy * y), slope = 2, I0=3.7) -y)/tau_y
    return [dx, dy]

#──── WILCO SINUSOIDALLY FORCED  ──────────────────────────────────────────────────────────────
def wilco_forced(t, state, tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, A, f, noise_std=0.0): #sinusoidal force
    x, y = state
    noise = noise_std * np.random.randn()  # additive noise on E
    F_e = A * np.sin(2 * np.pi * f *t/1000)
    dx = (sigmoid((w_xx * x - w_xy * y + P_x + F_e + noise), slope =1.3, I0=4.0)
          -x)/tau_x
    dy = (sigmoid((w_yx * x - w_yy * y), slope = 2, I0=3.7) -y)/tau_y
    return [dx, dy]

#──── WILCO NETWORK  ──────────────────────────────────────────────────────────────
def wilco_network(t, state, tau_x, tau_y, w_xx, w_xy, w_yx, w_yy, P_x, G, C, A,f, noise_std=0.0):
    N = len(P_x)
    x = state[:N]
    y = state[N:]

    F_e = np.zeros(N)
    F_e[0] = A * np.sin(2 * np.pi * f *t /1000) #t in s not ms 

    coupling = G * (C @ x)
    noise = noise_std * np.random.randn(N)

    dx = (sigmoid((w_xx*x - w_xy*y + P_x + F_e + coupling + noise),slope =1.3, I0=4.0)
          -x)/tau_x
    dy = (sigmoid((w_yx*x -w_yy*y), slope =2.0, I0=3.7)
          -y)/tau_y

    return np.concatenate([dx, dy])

