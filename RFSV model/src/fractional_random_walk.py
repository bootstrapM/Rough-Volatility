import numpy as np
import matplotlib.pyplot as plt

class FractionalRandomWalk:
    def __init__(self, N, H):
        """
        Initialize the FractionalRandomWalk.
        
        Args:
            N (int): Number of time steps
            H (float): Hurst parameter, must be between 0 and 1
        """
        self.N = N
        self.H = H
        self.N_half = N // 2
        self.time_series = None
        self.alpha = None
        self.beta = None
        
    def generate_noise(self):
        """
        Generate the alpha and beta Gaussian variables with variance 1/k^(1+2H).
        """
        k = np.arange(1, self.N_half + 1)
        variances = 1 / (k ** (1 + 2 * self.H))
        
        # Generate the gaussian variables
        self.alpha = np.random.normal(0, np.sqrt(variances), self.N_half)   #Shape: (N_half,)
        self.beta = np.random.normal(0, np.sqrt(variances), self.N_half)    #Shape: (N_half,)
        
    def generate_eta_series(self):
        """
        Generate the time series eta_t by superposing Gaussian white noises.
        Vectorized implementation for better performance.
        
        Returns:
            numpy.ndarray: The generated time series
        """
        if self.alpha is None or self.beta is None:
            self.generate_noise()
            
        # Create time points and k values
        t = np.arange(1, self.N + 1)[:, np.newaxis]  # Shape: (N, 1)  ;; np.newaxis convert the shape (N,) to (N, 1)
        k = np.arange(1, self.N_half + 1)            # Shape: (N_half,)
        
        # Compute the phase terms for all t and k
        phase = 2 * np.pi * k * t / self.N           # Shape: (N, N_half)
        
        # Compute cosine and sine terms
        cos_terms = np.cos(phase)                    # Shape: (N, N_half)
        sin_terms = np.sin(phase)                    # Shape: (N, N_half)
        
        # Multiply by coefficients and sum
        eta = (cos_terms @ self.alpha + sin_terms @ self.beta) # Shape: (N,)
            
        self.time_series = eta
        return eta
    
    def compute_correlation(self, min_lag=0, max_lag=None):
        """
        Compute the correlation function C(t,s) analytically.
        
        Args:
            max_lag (int, optional): Maximum lag to compute. Defaults to N//2.
            
        Returns:
            tuple: (lags, correlation values)
        """
        if max_lag is None:
            max_lag = self.N // 2
            
        k = np.arange(1, self.N_half + 1)                   # Shape: (N_half,)
        lags = np.arange(min_lag, max_lag)[:, np.newaxis]   # Shape: (max_lag-min_lag, 1)
        
        phase = 2 * np.pi * lags * k / self.N           # Shape: (max_lag-min_lag, N_half)
        weights = 1 / (k ** (1 + 2 * self.H))           # Shape: (N_half,)
        corr = np.cos(phase) @ weights                  # Shape: (max_lag-min_lag,)
            
        return lags[:, 0], corr
    
    def compute_variogram(self, max_lag=None):
        """
        Compute the variogram V(t-s) = E[(eta_t - eta_s)^2].
        
        Args:
            max_lag (int, optional): Maximum lag to compute. Defaults to N//2.
            
        Returns:
            tuple: (lags, variogram values)
        """
        if max_lag is None:
            max_lag = self.N // 2
            
        lags, corr = self.compute_correlation(max_lag=max_lag)
        # Variogram is defined as V(l) = 2 * (C(0) - C(l)) for a stationary process
        variogram = 2 * (corr[0] - corr)
        
        return lags, variogram
    
    def compute_higher_moment_variogram(self, order=2, max_lag=None):
        """
        Compute higher-order variograms E[|eta_t - eta_s|^q] where q is the order.
        
        Args:
            order (float): Order of the variogram (q)
            max_lag (int, optional): Maximum lag to compute. Defaults to N//2.
            
        Returns:
            tuple: (lags, variogram values)
        """
        if max_lag is None:
            max_lag = self.N // 2
            
        if self.time_series is None:
            self.generate_time_series()
            
        lags = np.arange(max_lag)
        variogram = np.zeros(max_lag)
        
        # For each lag, compute the q-th moment of increments
        for l in lags:
            if l == 0:
                variogram[l] = 0
                continue
                
            # Compute increments for this lag
            increments = self.time_series[l:] - self.time_series[:-l]
            # Compute q-th moment
            variogram[l] = np.mean(np.abs(increments) ** order)
            
        return lags, variogram
        
    def analyze_scaling_exponents(self, orders=None, max_lag=None):
        """
        Analyze scaling exponents for different orders of variograms.
        For a fractional process, E[|eta_t - eta_s|^q] ~ |t-s|^zeta(q)
        where zeta(q) is the scaling exponent.
        
        Args:
            orders (list, optional): List of orders to analyze. Defaults to [1,2,3,4].
            max_lag (int, optional): Maximum lag to use. Defaults to N//4.
            
        Returns:
            tuple: (orders, scaling_exponents)
        """
        if orders is None:
            orders = [1, 2, 3, 4]
            
        if max_lag is None:
            max_lag = self.N // 4
            
        scaling_exponents = []
        
        plt.figure(figsize=(15, 6))
        
        # Plot for each order
        plt.subplot(1, 2, 1)
        for q in orders:
            lags, var = self.compute_higher_moment_variogram(order=q, max_lag=max_lag)
            
            # Remove zero lag and any zero values for log-log plot
            mask = (lags > 0) & (var > 0)
            lags = lags[mask]
            var = var[mask]
            
            # Fit power law in log-log space
            coeffs = np.polyfit(np.log(lags), np.log(var), 1)
            scaling_exponents.append(coeffs[0])
            
            # Plot variogram
            plt.loglog(lags, var, 'o-', label=f'q={q}')
            # Plot fit
            fit_line = np.exp(coeffs[1]) * lags**coeffs[0]
            plt.loglog(lags, fit_line, '--', alpha=0.5)
            
        plt.grid(True)
        plt.legend()
        plt.title(f'Higher-order Variograms (H = {self.H})')
        plt.xlabel('Lag')
        plt.ylabel('E[|Δη|^q]')
        
        # Plot scaling exponents
        plt.subplot(1, 2, 2)
        plt.plot(orders, scaling_exponents, 'o-')
        plt.plot(orders, [q * self.H for q in orders], '--', 
                label='Linear scaling')
        plt.grid(True)
        plt.legend()
        plt.title(f'Scaling Exponents ζ(q) (H = {self.H})')
        plt.xlabel('Order q')
        plt.ylabel('ζ(q)')
        
        plt.tight_layout()
        plt.show()
        
        return np.array(orders), np.array(scaling_exponents)
    
    def plot_time_series(self):
        """
        Plot the generated time series.
        """
        if self.time_series is None:
            self.generate_time_series()
            
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(self.N), self.time_series)
        plt.title(f'Fractional Random Walk (H = {self.H})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
        
    def plot_variogram(self, max_lag=None):
        """
        Plot the variogram and its theoretical scaling.
        """
        if max_lag is None:
            max_lag = self.N // 4
            
        lags, variogram = self.compute_variogram(max_lag)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(lags[1:], variogram[1:], 'b-', label='Empirical')
        
        # Plot theoretical scaling l^(2H)
        x = np.logspace(0, np.log10(max_lag), 100)
        y = x ** (2 * self.H) * variogram[1] # Scaled by variogram[1] to match the first empirical value thats at lag=1
        plt.loglog(x, y, 'r--', label=f'Theoretical (∝ l^{2*self.H:.2f})')
        
        plt.title(f'Variogram Analysis (H = {self.H})')
        plt.xlabel('Lag (l)')
        plt.ylabel('V(l)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_correlation_analysis(self, min_lag=0, max_lag=None):
        """
        Plot the analytical correlation function C(l) and analyze its dependence on l^(2H).
        
        Args:
            max_lag (int, optional): Maximum lag to compute. Defaults to N//2.
        """
        if max_lag is None:
            max_lag = self.N // 2
            
        lags, corr = self.compute_correlation(min_lag, max_lag)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: C(l) vs l=lags
        plt.subplot(1, 2, 1)
        plt.plot(lags, corr, 'b-', label='C(ℓ)')
        plt.title('Correlation Function')
        plt.xlabel('Lag (ℓ)')
        plt.ylabel('C(ℓ)')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: C(l) vs l^(2H) (should see a linear relationship for 1 << lag << N)
        plt.subplot(1, 2, 2)
        
        # Create l^(2H) values
        l_2h = lags ** (2 * self.H)
        
        plt.plot(l_2h, corr, 'bo', label='|C(ℓ)| vs ℓ^(2H)')
        
        coeffs = np.polyfit(l_2h, corr, 1)

        fit_line = np.poly1d(coeffs)
        fit_line_values = fit_line(l_2h)
                            
        plt.plot(l_2h, 
                fit_line_values, 
                'r--', 
                label=f'Fit: slope = {coeffs[0]:.6f}')
            
        plt.title('Correlation vs ℓ^(2H)')
        plt.xlabel('ℓ^(2H)')
        plt.ylabel('C(ℓ)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()