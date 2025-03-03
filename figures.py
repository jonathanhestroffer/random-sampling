import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sampler import RandomSampler
from toy_problem import Rastrigin

def scatter_figure(samplesList: list[np.ndarray], titleList: list[str]) -> plt.figure:
    """Scatterplot for multiple sample sets.

    Args:
        samples_list : list of generated samples [(n_samples, dim),...]
        titles_list  : list of plot titles [str,...].

    Returns:
        plt.figure
    """

    num_axes = len(samplesList)

    fig, axs = plt.subplots(figsize=(num_axes*10,10), ncols=num_axes)
    
    if num_axes == 1:
        axs = [axs]
    
    for ax, samples, title in zip(axs, samplesList, titleList):
        sns.scatterplot(x=samples[:,0], y=samples[:,1], ax=ax)
        ax.set_xlabel(r"$x_{1}$")
        ax.set_ylabel(r"$x_{2}$", rotation=0, labelpad=20)
        ax.set_title(title)
        ax.set_xlim(-0.05,1.05)
        ax.set_ylim(-0.05,1.05)
        fig.tight_layout(pad=1.5)

    return fig

if __name__ == "__main__":
    
    sns.set_theme(style="ticks", font_scale=2.5, palette="muted")

    # Initialize RandomSampler
    Sampler = RandomSampler(rng=np.random.default_rng(seed=1))

    # Generate samples
    mc_samples = Sampler.sample(3, 128, "MC")
    lhs_samples = Sampler.sample(3, 121, "LHS", strength=2)
    sobol_samples = Sampler.sample(3, 128, "Sobol", scramble=False)
    halton_samples = Sampler.sample(3, 128, "Halton", scramble=False)
    r_sobol_samples = Sampler.sample(3, 128, "Sobol", scramble=True)
    r_halton_samples = Sampler.sample(3, 128, "Halton", scramble=True)


    #================#
    #    2D Plots    #
    #================#

    # MC v LHS figure
    fig = scatter_figure(
        samplesList=[mc_samples, lhs_samples], 
        titleList=[f"Monte Carlo, N = 128", "Latin Hypercube (orthogonal), N = 121"]
    )
    fig.savefig("./figures/mc_lhs_2d.png", dpi=300, bbox_inches="tight")
    
    # QMC Figure
    fig = scatter_figure(
        samplesList=[sobol_samples, halton_samples],
        titleList=[f"Sobol, N = 128", f"Halton, N = 128"]
    )
    fig.savefig("./figures/qmc_2d.png", dpi=300, bbox_inches="tight")

    # RQMC Figure
    fig = scatter_figure(
        samplesList=[r_sobol_samples, r_halton_samples],
        titleList=[f"Sobol (random), N = 128", f"Halton (random), N = 128"]
    )
    fig.savefig("./figures/rqmc_2d.png", dpi=300, bbox_inches="tight")


    #===================#
    #    Toy Problem    #
    #===================#

    # Rastrigin figure
    fig, ax = plt.subplots(figsize=(10,10))

    # Set up 2D grid 
    grid = np.linspace(-5.12, 5.12, 1000)
    X, Y = np.meshgrid(grid,grid)
    x = np.array([X.ravel(),Y.ravel()]).T
    Z = Rastrigin(x).reshape(X.shape)

    im = ax.pcolormesh(X, Y, Z, cmap="jet")
    ax.contour(X, Y, Z, colors="white", alpha=0.5)

    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$", rotation=0, labelpad=20)
    ax.set_title("2D Rastrigin Function")

    cax = fig.add_axes([1, 0.11158333333333337, 0.05, 0.829])
    cbar = plt.colorbar(im, cax=cax)

    fig.tight_layout(pad=0.5)
    plt.savefig("./figures/rastrigin.png", dpi=300, bbox_inches="tight")

    
    # Load saved data
    df_max = pd.read_pickle("./data/maximize_df.pkl")
    df_min = pd.read_pickle("./data/minimize_df.pkl")

    # Normalize max values
    # max f(x) = dim * 40.35329019
    df_max["Best Value"] = df_max.apply(lambda x: x["Best Value"] / (x["Dimensions"]*40.35329019), axis=1)

    # Optimization figure
    fig, axs  = plt.subplots(figsize=(18,9),ncols=2)
    sns.boxplot(data=df_min, x="Dimensions", y="Best Value", hue="Sampling Method", ax=axs[0])
    sns.boxplot(data=df_max, x="Dimensions", y="Best Value", hue="Sampling Method", ax=axs[1], legend=False)
    axs[0].set_title(r"Minimize $f(\mathbf{x}$)")
    axs[0].set_ylabel("Best Value")
    axs[1].set_title(r"Maximize $f(\mathbf{x}$)")
    axs[1].set_ylabel("Normalized Best Value")
    fig.tight_layout(pad=1.5)
    fig.savefig("./figures/optimization.png", dpi=300, bbox_inches="tight")
