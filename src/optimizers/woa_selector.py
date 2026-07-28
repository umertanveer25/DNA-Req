# src/optimizers/woa_selector.py
"""Whale Optimization Algorithm (WOA) feature selector.

Whales either encircle prey (exploitation), search randomly
(exploration), or perform a spiral bubble-net attack.  A sigmoid
transfer function converts continuous positions to binary masks.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


class WOAFeatureSelector:
    def __init__(self, pod_size=30, iterations=10, b=1.0, random_state=42):
        self.pod_size = pod_size
        self.iterations = iterations
        self.b = b                  # spiral shape constant
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
        """Run WOA and return a boolean mask of selected features."""
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        # Initialise continuous positions in [−1, 1]
        positions = [np.random.uniform(-1, 1, n_features)
                     for _ in range(self.pod_size)]

        def to_mask(pos):
            prob = self._sigmoid(pos)
            m = (np.random.rand(n_features) < prob).astype(bool)
            if not np.any(m):
                m[np.random.randint(n_features)] = True
            return m

        fitnesses = [self._fitness(to_mask(p), X, y) for p in positions]
        best_idx = int(np.argmax(fitnesses))
        best_pos = positions[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        best_mask = to_mask(best_pos)

        for t in range(self.iterations):
            a = 2 - 2 * t / self.iterations  # linearly decreases 2→0

            for i in range(self.pod_size):
                p = np.random.rand()
                r = np.random.rand(n_features)
                A = 2 * a * r - a
                C = 2 * np.random.rand(n_features)

                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # Encircling prey (exploitation)
                        D = np.abs(C * best_pos - positions[i])
                        positions[i] = best_pos - A * D
                    else:
                        # Search for prey (exploration)
                        rand_idx = np.random.randint(self.pod_size)
                        D = np.abs(C * positions[rand_idx] - positions[i])
                        positions[i] = positions[rand_idx] - A * D
                else:
                    # Spiral bubble-net attack
                    D_prime = np.abs(best_pos - positions[i])
                    l = np.random.uniform(-1, 1, n_features)
                    positions[i] = (D_prime * np.exp(self.b * l)
                                    * np.cos(2 * np.pi * l) + best_pos)

                # Evaluate
                mask = to_mask(positions[i])
                f = self._fitness(mask, X, y)
                fitnesses[i] = f
                if f > best_fitness:
                    best_fitness = f
                    best_pos = positions[i].copy()
                    best_mask = mask.copy()

        if best_mask is None or not np.any(best_mask):
            return np.ones(n_features, dtype=bool)
        return best_mask
