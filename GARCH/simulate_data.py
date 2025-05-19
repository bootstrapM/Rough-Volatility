import numpy as np
import pandas as pd
from typing import Tuple

def simulate_garch_process(
    n_samples: int = 1000,
    omega: float = 0.1,
    alpha: float = 0.2,
    beta: float = 0.7,
    random_state: int = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate a GARCH(1,1) process with specified parameters.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    omega : float
        Constant term in variance equation
    alpha : float
        ARCH parameter (impact of past squared returns)
    beta : float
        GARCH parameter (persistence of volatility)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    returns : pd.Series
        Simulated returns
    volatility : pd.Series
        True volatility process
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Initialize arrays
    returns = np.zeros(n_samples)
    sigma2 = np.zeros(n_samples)
    
    # Set initial variance
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate the process
    for t in range(1, n_samples):
        # Generate random shock
        z = np.random.standard_normal()
        
        # Calculate return
        returns[t] = np.sqrt(sigma2[t-1]) * z
        
        # Update conditional variance
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Convert to pandas Series with datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    returns = pd.Series(returns, index=dates, name='returns')
    volatility = pd.Series(np.sqrt(sigma2), index=dates, name='volatility')
    
    return returns, volatility

def simulate_garch_process_with_shock(
    n_samples: int = 1000,
    omega: float = 0.1,
    alpha: float = 0.2,
    beta: float = 0.7,
    shock_time: int = 500,  # When the shock occurs
    shock_magnitude: float = 5.0,  # Multiplier for the shock
    random_state: int = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate a GARCH(1,1) process with a volatility shock at a specified time.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    omega : float
        Constant term in variance equation
    alpha : float
        ARCH parameter (impact of past squared returns)
    beta : float
        GARCH parameter (persistence of volatility)
    shock_time : int
        Time index when the shock occurs
    shock_magnitude : float
        Magnitude of the shock (multiplier for the return)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    returns : pd.Series
        Simulated returns
    volatility : pd.Series
        True volatility process
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Initialize arrays
    returns = np.zeros(n_samples)
    sigma2 = np.zeros(n_samples)
    
    # Set initial variance
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate the process
    for t in range(1, n_samples):
        # Generate random shock
        z = np.random.standard_normal()
        
        # Add the shock at specified time
        if t == shock_time:
            z *= shock_magnitude
        
        # Calculate return
        returns[t] = np.sqrt(sigma2[t-1]) * z
        
        # Update conditional variance
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Convert to pandas Series with datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    returns = pd.Series(returns, index=dates, name='returns')
    volatility = pd.Series(np.sqrt(sigma2), index=dates, name='volatility')
    
    return returns, volatility

def simulate_homoskedastic_process(
    n_samples: int = 1000,
    mu: float = 0.0,
    sigma: float = 0.02,
    random_state: int = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate a homoskedastic process (constant volatility) with normal returns.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    mu : float
        Mean of returns
    sigma : float
        Constant volatility (standard deviation)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    returns : pd.Series
        Simulated returns
    volatility : pd.Series
        Constant volatility series
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate normal returns with constant volatility
    returns = np.random.normal(mu, sigma, n_samples)
    
    # Create constant volatility series
    volatility = np.full(n_samples, sigma)
    
    # Convert to pandas Series with datetime index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    returns = pd.Series(returns, index=dates, name='returns')
    volatility = pd.Series(volatility, index=dates, name='volatility')
    
    return returns, volatility

def add_price_series(returns: pd.Series, initial_price: float = 100) -> pd.Series:
    """
    Convert returns to price series.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    initial_price : float
        Starting price level
        
    Returns:
    --------
    pd.Series
        Price series
    """
    return initial_price * (1 + returns).cumprod()

def generate_sample_data(
    n_samples: int = 1000,
    garch_params: dict = None,
    random_state: int = None,
    homoskedastic: bool = False,
    add_shock: bool = False,
    shock_params: dict = None
) -> pd.DataFrame:
    """
    Generate a complete sample dataset with returns, volatility, and prices.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    garch_params : dict
        Dictionary of GARCH parameters (omega, alpha, beta)
    random_state : int
        Random seed for reproducibility
    homoskedastic : bool
        If True, generate homoskedastic data instead of GARCH process
    add_shock : bool
        If True, add a volatility shock to the GARCH process
    shock_params : dict
        Dictionary containing shock parameters (time, magnitude)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing returns, volatility, and prices
    """
    if homoskedastic:
        # Generate homoskedastic data
        returns, volatility = simulate_homoskedastic_process(
            n_samples=n_samples,
            mu=0.0,
            sigma=0.02,  # Constant volatility of 2%
            random_state=random_state
        )
    else:
        # Generate GARCH process
        if garch_params is None:
            garch_params = {'omega': 0.1, 'alpha': 0.2, 'beta': 0.7}
            
        if add_shock:
            if shock_params is None:
                shock_params = {'time': n_samples // 2, 'magnitude': 5.0}
                
            returns, volatility = simulate_garch_process_with_shock(
                n_samples=n_samples,
                omega=garch_params['omega'],
                alpha=garch_params['alpha'],
                beta=garch_params['beta'],
                shock_time=shock_params['time'],
                shock_magnitude=shock_params['magnitude'],
                random_state=random_state
            )
        else:
            returns, volatility = simulate_garch_process(
                n_samples=n_samples,
                omega=garch_params['omega'],
                alpha=garch_params['alpha'],
                beta=garch_params['beta'],
                random_state=random_state
            )
    
    # Generate price series
    prices = add_price_series(returns)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'returns': returns,
        'volatility': volatility,
        'prices': prices
    })
    
    return df 