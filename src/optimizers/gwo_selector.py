# src/optimizers/gwo_selector.py
"""Grey Wolf Optimizer (GWO) feature selector.

The pack is led by Alpha (best), Beta (2nd best), and Delta (3rd best).
Omega wolves update their positions by averaging the influence of the
three leaders.  A sigmoid transfer function converts continuous
positions to binary feature masks.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


class GWOFeatureSelector:
    def __init__(self, pack_size=30, iterations=10, random_state=42):
        self.pack_size = pack_size
        self.iterations = iterations
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

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def select_features(self, X, y):
        """Run GWO and return a boolean mask of selected features."""
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        # Initialise continuous positions in [−1, 1]
        positions = [np.random.uniform(-1, 1, n_features)
                     for _ in range(self.pack_size)]

        # Convert to binary and evaluate
        def to_mask(pos):
            prob = self._sigmoid(pos)
            m = (np.random.rand(n_features) < prob).astype(bool)
            if not np.any(m):
                m[np.random.randint(n_features)] = True
            return m

        fitnesses = [self._fitness(to_mask(p), X, y) for p in positions]

        # Identify Alpha, Beta, Delta
        ranked = np.argsort(fitnesses)[::-1]
        alpha_pos = positions[ranked[0]].copy()
        alpha_score = fitnesses[ranked[0]]
        beta_pos = positions[ranked[1]].copy()
        delta_pos = positions[ranked[2]].copy()

        best_mask = to_mask(alpha_pos)
        best_fitness = alpha_score

        for t in range(self.iterations):
            a = 2 - 2 * t / self.iterations  # linearly decreases from 2→0

            for i in range(self.pack_size):
                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - positions[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - positions[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - positions[i])
                X3 = delta_pos - A3 * D_delta

                positions[i] = (X1 + X2 + X3) / 3.0

            # Evaluate new positions
            fitnesses = [self._fitness(to_mask(p), X, y) for p in positions]
            ranked = np.argsort(fitnesses)[::-1]
            alpha_pos = positions[ranked[0]].copy()
            beta_pos = positions[ranked[1]].copy()
            delta_pos = positions[ranked[2]].copy()

            if fitnesses[ranked[0]] > best_fitness:
                best_fitness = fitnesses[ranked[0]]
                best_mask = to_mask(alpha_pos)

        if best_mask is None or not np.any(best_mask):
            return np.ones(n_features, dtype=bool)
        return best_mask
