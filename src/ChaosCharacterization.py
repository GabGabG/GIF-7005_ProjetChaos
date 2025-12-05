import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class TentMap:

    def __init__(self, timeseries: np.ndarray, dim: int = 0):
        # La série temporelle doit être de shape (n temps, n features)
        self.timeseries = timeseries[..., dim]
        self.tent_map = None

    def compute_tent_map(self, maximums_only: bool = False, *args, **kwargs):
        indices, _ = find_peaks(self.timeseries, *args, **kwargs)
        if not maximums_only:
            # Minimums -> maximums lorsque x -> -x
            min_indices, _ = find_peaks(-self.timeseries, *args, **kwargs)
            # On trie les indices pour les garder en ordre croissant, très important
            indices = np.sort(np.concatenate([indices, min_indices]))
        xn_i = indices[:-1]
        xnp1_i = indices[1:]
        xn = self.timeseries[xn_i]
        xnp1 = self.timeseries[xnp1_i]
        self.tent_map = xn, xnp1

    def display_tent_map(self, ax: plt.Axes = None, *scatter_args,
                         **scatter_kwargs):
        if self.tent_map is None:
            raise ValueError("Veuillez calculer les sommets de l'application avec `compute_tent_map`.")
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(*self.tent_map, *scatter_args, **scatter_kwargs)
        return ax


