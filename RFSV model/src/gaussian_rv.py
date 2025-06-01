import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class GaussianRandomVariables:
    def __init__(self, N):
        """
        Initialize the GaussianRandomVariables generator.
        
        Args:
            N (int): The number of pairs of random variables. Total variables (N) will be split into two equal sets.
        """
        self.N = N
        self.N_half = N // 2
    
    def generate_variables(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate alpha_k and beta_k variables with variance 1/k^2.
        
        Returns:
            tuple: (alpha, beta) arrays of random variables
        """
        k = np.arange(1, self.N_half + 1)
        variances = 1 / (k * k)
        
        # Generate the required standard normal variables and store them in alpha and beta
        alpha = np.random.normal(0, np.sqrt(variances), self.N_half)
        beta = np.random.normal(0, np.sqrt(variances), self.N_half)
        
        return alpha, beta
    
    def calculate_eta(self) -> float:
        """
        Calculate η as the sum of alpha_k and beta_k.
        
        Returns:
            float: The value of η
        """
        alpha, beta = self.generate_variables()
        eta = np.sum(alpha) + np.sum(beta)
        return eta
    
    def theoretical_variance(self):
        """
        Calculate the theoretical variance of η.
        
        Returns:
            float: Theoretical variance
        """
        k = np.arange(1, self.N_half + 1)
        return 2 * np.sum(1 / (k * k))
    
    def empirical_variance(self, num_samples=10000):
        """
        Calculate the empirical variance of η over a number of samples (default is 10000)
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            float: Empirical variance
        """
        samples = np.array([self.calculate_eta() for _ in range(num_samples)])
        return np.var(samples)
    
    def get_variance(self, num_samples=10000):
        """
        Returns the theoretical and empirical variance of η.
        
        Args:
            num_samples (int): Number of samples for empirical calculation
            
        Returns:
            tuple: (theoretical_variance, empirical_variance)
        """
        theo_var = self.theoretical_variance()
        emp_var = self.empirical_variance(num_samples)
        
        return theo_var, emp_var
    
    def plot_distribution(self, num_samples=10000):
        """
        Plot the distribution of η values.
        
        Args:
            num_samples (int): Number of samples to generate
        """
        samples = np.array([self.calculate_eta() for _ in range(num_samples)])
        
        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins=50, density=True, alpha=0.7, color='blue')
        plt.title(f'Distribution of η (N={self.N})')
        plt.xlabel('η values')
        plt.ylabel('Frequency')
        
        # Add theoretical normal distribution for comparison
        x = np.linspace(min(samples), max(samples), 100)
        theo_var = self.theoretical_variance()
        plt.plot(x, 
                1/np.sqrt(2*np.pi*theo_var) * np.exp(-x**2/(2*theo_var)), 
                'r-', 
                lw=2, 
                label='Theoretical Normal')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # An example 
    N = 100 
    grv = GaussianRandomVariables(N)
    
    # Variance calculation
    theo_var, emp_var = grv.get_variance(50000)
    print(f"Theoretical variance: {theo_var:.6f}")
    print(f"Empirical variance: {emp_var:.6f}")
    print(f"Relative difference: {abs(theo_var - emp_var)/theo_var*100:.2f}%")
    
    # distribution of eta
    grv.plot_distribution() 