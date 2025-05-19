from arch.__future__ import reindexing
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple, Dict
import warnings

class GARCHAnalyzer:
    def __init__(self, returns: pd.Series):
        """
        Initialize GARCH analyzer with return series.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns to analyze
        """
        self.returns = returns
        self.model = None
        self.results = None
        
    def fit_model(self, p: int = 1, q: int = 1) -> Dict:
        """
        Fit GARCH(p,q) model to the return series.
        
        Parameters:
        -----------
        p : int
            Order of ARCH term
        q : int
            Order of GARCH term
            
        Returns:
        --------
        Dict
            Dictionary containing model parameters and statistics
        """
        # Suppress warnings for convergence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fit the model
            self.model = arch_model(
                self.returns,
                vol='Garch',
                p=p,
                q=q,
                mean='Zero',
                dist='normal'
            )
            self.results = self.model.fit(disp='off')
            
        # Extract parameters
        params = self.results.params
        
        # Calculate persistence
        persistence = sum(params[1:])
        
        # Create summary dictionary
        summary = {
            'omega': params['omega'],
            'alpha': params[f'alpha[1]'],
            'beta': params[f'beta[1]'],
            'persistence': persistence,
            'log_likelihood': self.results.loglikelihood,
            'aic': self.results.aic,
            'bic': self.results.bic
        }
        
        return summary
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Get the estimated conditional volatility from the fitted model.
        
        Returns:
        --------
        pd.Series
            Conditional volatility estimates
        """
        if self.results is None:
            raise ValueError("Model must be fit before getting conditional volatility")
            
        return pd.Series(
            np.sqrt(self.results.conditional_volatility),
            index=self.returns.index,
            name='conditional_volatility'
        )
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate various statistics for the return series.
        
        Returns:
        --------
        Dict
            Dictionary containing various statistics
        """
        stats = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'skewness': self.returns.skew(),
            'kurtosis': self.returns.kurtosis(),
            'min': self.returns.min(),
            'max': self.returns.max()
        }
        
        # Calculate autocorrelation of squared returns
        squared_returns = self.returns ** 2
        acf_squared = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0,1]
        stats['acf_squared_returns'] = acf_squared
        
        return stats 