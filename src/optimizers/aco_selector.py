# src/optimizers/aco_selector.py
"""Ant Colony Optimization (ACO) based feature selector.

Each ant constructs a binary feature mask by probabilistically selecting
features based on pheromone trails.  After evaluation, pheromone trails
are updated: evaporation + deposit proportional to fitness.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


class ACOFeatureSelector:
    def __init__(self, n_ants=30, iterations=10, alpha=1.0, rho=0.1,
                 random_state=42):
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha          # pheromone importance
        self.rho = rho              # evaporation rate
        self.random_state = random_state
        self.fitness_model = SVC(kernel="linear", C=1,
                                 random_state=self.random_state)

    def _fitness(self, mask, X, y):
        if not np.any(mask):
            return 0.0
        X_subset = X[:, mask]
        skf = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=self.random_state)
        scores = cross_val_score(self.fitness_model, X_subset, y,
                                 cv=skf, scoring="f1_macro")
        return np.mean(scores)

    def select_features(self, X, y):
        """Run ACO and return a boolean mask of selected features."""
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        # Initialise pheromone trails (equal for all features)
        tau = np.ones(n_features) * 0.5

        best_mask = None
        best_fitness = 0.0

        for _gen in range(self.iterations):
            masks = []
            fitnesses = []
            for _ in range(self.n_ants):
                # Selection probability based on pheromone
                tau_alpha = tau ** self.alpha
                prob = tau_alpha / (tau_alpha + (1 - tau) ** self.alpha + 1e-12)
                mask = (np.random.rand(n_features) < prob).astype(bool)
                # Ensure at least one feature
                if not np.any(mask):
                    mask[np.random.randint(n_features)] = True
                masks.append(mask)
                f = self._fitness(mask, X, y)
                fitnesses.append(f)

                if f > best_fitness:
                    best_fitness = f
                    best_mask = mask.copy()

            # Pheromone evaporation
            tau *= (1 - self.rho)

            # Pheromone deposit (proportional to fitness)
            for mask, f in zip(masks, fitnesses):
                tau[mask] += f * self.rho

            # Clamp pheromone in [0.01, 0.99]
            tau = np.clip(tau, 0.01, 0.99)

        if best_mask is None or not np.any(best_mask):
            return np.ones(n_features, dtype=bool)
        return best_mask
