import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional
from heston_model import HestonModel, HestonParameters

class HestonPricer:
    """
    Class for pricing European options under the Heston model.
    Implements both characteristic function and Monte Carlo methods.
    """
    
    def __init__(self, model: HestonModel): 
        self.model = model
        self.params = model.params
    
    def Heston_CF(self, u: float, t: float, r: float) -> complex:
        """
        Compute the characteristic function of the Heston model.
        
        Args:
            u: Complex number
            t: Time to maturity
            r: Risk-free rate
            
        Returns:
            Complex value of the characteristic function
        """
        # Model parameters
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        v0 = self.params.v0
        
        
        # Compute the terms for the characteristic function
        d = np.sqrt((rho * sigma * 1j * u - kappa)**2 + sigma**2 * (1j * u + u**2)) # this was written correctly! 
        g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d) # this was also written correctly! 
        
        # Compute the characteristic function
        C = ((kappa * theta / sigma**2) * 
             ((kappa - rho * sigma * 1j * u - d) * t - 
              2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))))
        
        D = ((kappa - rho * sigma * 1j * u - d) / sigma**2 * 
             ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))))
        
        
        return np.exp(C + D * v0 + 1j * u * r * t)
    
    def price_option_CF(self, S0: float, K: float, T: float, 
                                r: float, upper_limit: int, option_type: str = 'call') -> Tuple[float, float]:
        """
        Price a European call option using the characteristic function method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            
        Returns:
            Tuple of (option price, standard error)
        """
        def integrand1(u: float) -> float:
            """First integrand for the call price formula"""
            
            phi1 = self.Heston_CF(u - 1j, T, r)
            phi2 = self.Heston_CF(- 1j, T, r)
            return np.real(np.exp(-1j * u * np.log(K/S0)) /(1j * u) * phi1 / phi2)
        
        def integrand2(u: float) -> float:
            """Second integrand for the call price formula"""

            phi = self.Heston_CF(u, T, r)
            return np.real(np.exp(-1j * u * np.log(K/S0)) /(1j * u) * phi)
        
        P1, err1 = quad(integrand1, 0, upper_limit, limit=10000)
        P2, err2 = quad(integrand2, 0, upper_limit, limit=10000)


        # Calculate payoffs based on option type
        if option_type.lower() == 'call':
            # Compute the call price
            term1 = S0 * (0.5 + P1/np.pi)
            term2 = K * np.exp(-r * T) * (0.5 + P2/np.pi)
            price = term1 - term2

            error1  = S0 * err1/np.pi
            error2  = K * np.exp(-r * T) * err2/np.pi
            std_err = error1 - error2 

            return price, std_err

        elif option_type.lower() == 'put':
            # Compute the put price (can be obtained from put-call parity: put_price = call_price - S0 + K * np.exp(-r * T) )
            term1 = S0 * (0.5 - P1/np.pi)
            term2 = K * np.exp(-r * T) * (0.5 - P2/np.pi)
            price = - term1 + term2

            error1  = - S0 * err1/np.pi
            error2  = - K * np.exp(-r * T) * err2/np.pi
            std_err = - error1 + error2 

            return price, std_err
        
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

    
    def price_option_MC(self, S0: float, K: float, T: float, r: float,
                       option_type: str = 'call',
                       n_paths: int = 10000, n_steps: int = 100) -> Tuple[float, float]:
        """
        Price a European option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: Type of option ('call' or 'put')
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            
        Returns:
            Tuple of (option price, standard error)
        """
        # Create a temporary model with risk-neutral drift mu = r
        risk_neutral_params = HestonParameters(
            kappa=self.params.kappa,
            theta=self.params.theta,
            sigma=self.params.sigma,
            rho=self.params.rho,
            v0=self.params.v0,
            mu=r 
        )
        risk_neutral_model = HestonModel(risk_neutral_params)
        
        # Simulate price paths
        prices, _ = risk_neutral_model.simulate_paths(S0, T, n_steps, n_paths, return_vol=True)
        
        # Calculate payoffs based on option type
        if option_type.lower() == 'call':
            payoffs = np.maximum(prices[:, -1] - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - prices[:, -1], 0)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        return price, std_err

    def implied_volatility(self, market_price: float, S0: float, K: float, 
                          T: float, r: float, option_type: str = 'call') -> float:
        """
        Calculate the implied volatility for a given market price using the Black-Scholes model.
        
        Args:
            market_price: Observed market price of the option
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: Type of option ('call' or 'put')
            
        Returns:
            Implied volatility
        """
        def black_scholes_price(sigma: float) -> float:
            """Black-Scholes price for a given volatility"""
            d1 = (np.log(S0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        
        def objective(sigma: float) -> float:
            """Objective function to minimize"""
            return (black_scholes_price(sigma) - market_price)**2
        
        # Find implied volatility using optimization
        result = minimize(objective, x0=0.3, bounds=[(0.01, 5.0)])
        
        if not result.success:
            print(f"Warning: Implied volatility optimization did not converge. Message: {result.message}")
        
        return result.x[0]

class BlackScholesPricer:
    """
    Class for pricing European options under the Black-Scholes model.
    Implements both closed-form and Monte Carlo methods.
    """
    
    def __init__(self, sigma: float):
        """
        Initialize the Black-Scholes pricer.
        
        Args:
            sigma: Volatility
        """
        self.sigma = sigma
    
    def price_call_closed(self, S0: float, K: float, T: float, r: float) -> float:
        """
        Price a European call option using the Black-Scholes closed-form formula.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            
        Returns:
            Option price
        """
        d1 = (np.log(S0/K) + (r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def price_put_closed(self, S0: float, K: float, T: float, r: float) -> float:
        """
        Price a European put option using the Black-Scholes closed-form formula.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            
        Returns:
            Option price
        """
        d1 = (np.log(S0/K) + (r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    def price_option_MC(self, S0: float, K: float, T: float, r: float,
                       option_type: str = 'call', n_paths: int = 10000,
                       n_steps: int = 100) -> Tuple[float, float]:
        """
        Price a European option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: Type of option ('call' or 'put')
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            
        Returns:
            Tuple of (option price, standard error)
        """
        # Time steps
        dt = T / n_steps
        
        # Vectorized code for generating random numbers
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Calculate drift and diffusion terms
        drift = (r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        # Calculate cumulative returns for all paths at once
        returns = drift + diffusion * Z
        cumulative_returns = np.cumsum(returns, axis=1)
        
        # Calculate final stock prices
        S = S0 * np.exp(cumulative_returns)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:  # put
            payoffs = np.maximum(K - S[:, -1], 0)
        
        # Discount payoffs
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        return price, std_err

def test_pricing():
    """Test the pricing methods with example parameters"""
    # Set up model parameters
    params = HestonParameters(
        kappa=1.2,    # Mean reversion
        theta=0.10,   # Long-term variance
        sigma=0.3,    # Volatility of variance
        rho=-0.8,     # Correlation
        v0=0.10,      # Initial variance
        mu=0.1        # Drift rate
    )
    
    # Create model and pricer
    model = HestonModel(params)
    pricer = HestonPricer(model)
    
    # Test parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity
    r = 0.05    # Risk-free rate
    
    print("\nModel Parameters:")
    print(f"kappa (mean reversion): {params.kappa}")
    print(f"theta (long-term variance): {params.theta}")
    print(f"sigma (vol of vol): {params.sigma}")
    print(f"rho (correlation): {params.rho}")
    print(f"v0 (initial variance): {params.v0}")
    print(f"mu (drift): {params.mu}")
    
    print("\nOption Parameters:")
    print(f"S0 (spot price): {S0}")
    print(f"K (strike price): {K}")
    print(f"T (time to maturity): {T}")
    print(f"r (risk-free rate): {r}")
    
    # Price using Heston characteristic function method
    print("\nPricing with Heston Characteristic Function Method...")
    call_price_cf, call_err_cf = pricer.price_option_CF(S0, K, T, r, 10000, 'call')
    put_price_cf, put_err_cf = pricer.price_option_CF(S0, K, T, r, 10000, 'put')
    print(f"Call Price: {call_price_cf:.4f} ± {call_err_cf:.4f}")
    print(f"Put Price: {put_price_cf:.4f} ± {put_err_cf:.4f}")
    
    # Calculate implied volatility from Heston prices
    iv_call = pricer.implied_volatility(call_price_cf, S0, K, T, r, 'call')
    iv_put = pricer.implied_volatility(put_price_cf, S0, K, T, r, 'put')
    print(f"\nImplied Volatilities from Heston Prices:")
    print(f"Call IV: {iv_call:.4f}")
    print(f"Put IV: {iv_put:.4f}")
    
    # Create Black-Scholes pricer using implied volatility from Heston
    bs_pricer_call = BlackScholesPricer(sigma=iv_call)
    bs_pricer_put = BlackScholesPricer(sigma=iv_put)
    
    # Price using Black-Scholes closed form with implied volatility
    print("\nPricing with Black-Scholes Closed Form (using Heston IV)...")
    bs_call_price = bs_pricer_call.price_call_closed(S0, K, T, r)
    bs_put_price = bs_pricer_put.price_put_closed(S0, K, T, r)
    print(f"Call Price (using Call IV): {bs_call_price:.4f}")
    print(f"Put Price (using Put IV): {bs_put_price:.4f}")
    
    # Price using Monte Carlo with different numbers of paths
    print("\nPricing with Monte Carlo Method...")
    n_paths_list = [10000, 50000, 100000]
    for n_paths in n_paths_list:
        # Heston prices
        call_price_mc, call_err_mc = pricer.price_option_MC(S0, K, T, r, 'call', n_paths=n_paths, n_steps=200)
        put_price_mc, put_err_mc = pricer.price_option_MC(S0, K, T, r, 'put', n_paths=n_paths, n_steps=200)
        print(f"\nHeston MC Prices ({n_paths} paths):")
        print(f"Call: {call_price_mc:.4f} ± {call_err_mc:.4f}")
        print(f"Put: {put_price_mc:.4f} ± {put_err_mc:.4f}")
        
        # Black-Scholes MC prices using implied volatility
        bs_call_mc, bs_call_err = bs_pricer_call.price_option_MC(S0, K, T, r, 'call', n_paths=n_paths, n_steps=200)
        bs_put_mc, bs_put_err = bs_pricer_put.price_option_MC(S0, K, T, r, 'put', n_paths=n_paths, n_steps=200)
        print(f"\nBlack-Scholes MC Prices ({n_paths} paths):")
        print(f"Call: {bs_call_mc:.4f} ± {bs_call_err:.4f}")
        print(f"Put: {bs_put_mc:.4f} ± {bs_put_err:.4f}")
    
    # Calculate percentage differences
    call_diff = abs(call_price_cf - bs_call_price) / bs_call_price * 100
    put_diff = abs(put_price_cf - bs_put_price) / bs_put_price * 100
    print(f"\nPercentage differences between Heston and Black-Scholes:")
    print(f"Call: {call_diff:.2f}%")
    print(f"Put: {put_diff:.2f}%")
    
    # Verify put-call parity
    print("\nPut-Call Parity Check:")
    print(f"Heston: Call - Put = {call_price_cf - put_price_cf:.4f}")
    print(f"Black-Scholes: Call - Put = {bs_call_price - bs_put_price:.4f}")
    print(f"Theoretical: S0 - K*exp(-rT) = {S0 - K * np.exp(-r * T):.4f}")

if __name__ == "__main__":
    test_pricing() 