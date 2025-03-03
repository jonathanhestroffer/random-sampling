from collections import defaultdict

import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
from optuna.distributions import FloatDistribution
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(verbosity=0)

from sampler import RandomSampler

def Rastrigin(x: np.ndarray) -> np.ndarray | float:
    """n-dimensional Rastrigin function.

    Ref: Rastrigin, L. A. "Systems of extremal control." Mir, Moscow (1974).

    Args:
        x : (n_samples, dim) array of points.
    
    Returns:
        np.ndarray | float : value of Rastrigin function evaluated at x.
    """
    f, n = 0, x.shape[1]
    for i in range(n):
        f += x[:,i]**2 - 10*np.cos(2*np.pi*x[:,i])

    f += 10*n

    if x.shape[0] == 1:
        return f.item()
    else:
        return f

def add_trials(study: optuna.Study, points: np.ndarray) -> None:
    """Add trials to Optuna study.

    Args:
        study   : Optuna study.
        points  : (n_samples, dim) array of points to add as initial trials.
    """
    for point in points:
        params, dists = {}, {}
        for i in range(dim):
            params[f"x_{i}"] = point[i]
            dists[f"x_{i}"] = FloatDistribution(-5.12, 5.12)

        trial = optuna.trial.create_trial(
            params = params,
            distributions = dists,
            value = Rastrigin(np.array([point]))
        )
        study.add_trial(trial)

if __name__ == "__main__":

    directions = ["minimize","maximize"]
    dimensions = [2, 4, 6, 8]
    n_inits = 100

    for direction in directions:
    
        dataDict = defaultdict(list)

        for dim in dimensions:
            
            print(f"Dimension: {dim}")

            def objective(trial: optuna.Trial):
                """Suggest and evaluate new point in sample space.
                """
                # suggest point in sample space
                point = np.empty((1, dim))
                for i in range(dim):
                    point[0,i] = trial.suggest_float(f"x_{i}", -5.12, 5.12)

                return Rastrigin(point)

            # iterate through initiatlization rounds
            for i in tqdm(range(n_inits), desc="Initialization rounds..."):
                
                # maintain roughly equivalent point density 
                # over total hypervolume.
                n_samples = 2**(3+dim)

                # initialize RandomSampler, new seed each run
                Sampler = RandomSampler(rng=np.random.default_rng(seed=i))

                # scale points to Rastrigin bounds
                mc_points = -5.12 + (2*5.12)*Sampler.sample(dim, n_samples, "MC")
                rqmc_points = -5.12 + (2*5.12)*Sampler.sample(dim, n_samples, "Sobol", scramble=True)

                # create studies, add initial trials, optimize
                mc_study = optuna.create_study(sampler=TPESampler(), direction=direction)   
                rqmc_study = optuna.create_study(sampler=TPESampler(), direction=direction)   

                add_trials(mc_study, mc_points)
                add_trials(rqmc_study, rqmc_points)

                mc_study.optimize(objective, n_trials=10, show_progress_bar=False)
                rqmc_study.optimize(objective, n_trials=10, show_progress_bar=False)

                # record best MC value
                dataDict["Sampling Method"].append("MC")
                dataDict["Dimension"].append(dim)
                dataDict["Best Value"].append(mc_study.best_value)

                # record best RQMC value
                dataDict["Sampling Method"].append("RQMC")
                dataDict["Dimension"].append(dim)
                dataDict["Best Value"].append(rqmc_study.best_value)
            
        # save as DataFrame
        df = pd.DataFrame.from_dict(dataDict)
        pd.to_pickle(df, f'./data/{direction}_df.pkl')