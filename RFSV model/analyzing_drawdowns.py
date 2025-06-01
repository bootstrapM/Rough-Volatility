import numpy as np
import matplotlib.pyplot as plt

def simulate_frsv_drawdown(eta, H=0.1, T=1.0, N=1000):
    dt = T / N
    t = np.linspace(0, T, N)
    # Generate rough volatility using fractional kernel
    K = lambda tau: tau**(H - 0.5)
    increments = np.random.normal(0, np.sqrt(dt), N)
    kernel = K(np.flip(t) + 1e-8)
    X = eta * np.convolve(increments, kernel, mode='full')[:N] * dt
    sigma = np.exp(X)
    # Simulate asset price
    dW = np.random.normal(0, np.sqrt(dt), N)
    S = np.zeros(N)
    S[0] = 100
    for i in range(1, N):
        S[i] = S[i-1] * (1 + sigma[i-1] * dW[i])
    max_S = np.maximum.accumulate(S)
    drawdown = (max_S - S) / max_S
    max_drawdown = np.max(drawdown)
    return max_drawdown

etas = np.linspace(0.1, 1.0, 10)
drawdowns = [simulate_frsv_drawdown(e) for e in etas]

plt.plot(etas, drawdowns)
plt.xlabel('Vol-of-Vol η')
plt.ylabel('Max Drawdown')
plt.title('Drawdown vs Vol-of-Vol in FRSV Model')
plt.grid(True)
plt.show()
