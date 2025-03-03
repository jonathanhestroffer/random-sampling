import numpy as np
from scipy.stats import qmc

class RandomSampler:
    """Collection of random sampling strategies including:
        
        - Monte Carlo (MC)
        - Latin Hypercube Sampling (LHS)
        - Quasi-Monte Carlo (Sobol, Halton)
    """
    def __init__(self, rng: np.random.Generator = np.random.default_rng()):
        """ Initialize random sampling strategy.

        Args:
            rng: NumPy random number generator.
        """
        self.rng = rng
        
    def sample(self, dim: int, n_samples: int, method: str, **kwargs) -> np.ndarray:
        """Generate random samples in range [0,1]

        Supports kwargs for scipy.stats.qmc engines
        
        Args:
            dim         : dimension.
            n_samples   : number of samples.
            method      : random sampling strategy/engine.

        Returns:
            np.ndarray  : (n_samples, dim) array of samples.
        """

        if method == "MC":
            return self.rng.random((n_samples, dim))
        elif method == "LHS":
            sampler = qmc.LatinHypercube(d=dim, rng=self.rng, **kwargs)
        elif method == "Sobol":
            sampler = qmc.Sobol(d=dim, rng=self.rng, **kwargs)
        elif method == "Halton":
            sampler = qmc.Halton(d=dim, rng=self.rng, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}. Choose from 'MC', 'LHS', 'Sobol', or 'Halton'")

        return sampler.random(n_samples)