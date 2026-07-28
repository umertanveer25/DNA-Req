# src/optimizers/abc_selector.py
"""Artificial Bee Colony (ABC) feature selector.

The colony is split into employed bees, onlooker bees, and scouts.
Employed bees exploit current food sources; onlookers choose sources
proportional to fitness; scouts replace stagnant sources.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


class ABCFeatureSelector:
    def __init__(self, colony_size=30, iterations=10, limit=5,
                 random_state=42):
        self.colony_size = colony_size
        self.iterations = iterations
        self.limit = limit          # stagnation limit before scout
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

    def _neighbour(self, source, n_features):
        """Generate a neighbour by flipping a few random bits."""
        neighbour = source.copy()
        n_flips = max(1, int(0.1 * n_features))
        flip_idx = np.random.choice(n_features, n_flips, replace=False)
        neighbour[flip_idx] = 1 - neighbour[flip_idx]
        if not np.any(neighbour):
            neighbour[np.random.randint(n_features)] = 1
        return neighbour

    def select_features(self, X, y):
        """Run ABC and return a boolean mask of selected features."""
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        n_food = self.colony_size // 2  # number of food sources

        # Initialise food sources as random binary masks
        sources = [np.random.randint(2, size=n_features) for _ in range(n_food)]
        fitnesses = [self._fitness(s.astype(bool), X, y) for s in sources]
        trials = [0] * n_food  # stagnation counters

        best_mask = None
        best_fitness = 0.0
        for i in range(n_food):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_mask = sources[i].copy()

        for _gen in range(self.iterations):
            # ── Employed bees phase ──
            for i in range(n_food):
                neighbour = self._neighbour(sources[i], n_features)
                f = self._fitness(neighbour.astype(bool), X, y)
                if f > fitnesses[i]:
                    sources[i] = neighbour
                    fitnesses[i] = f
                    trials[i] = 0
                else:
                    trials[i] += 1
                if fitnesses[i] > best_fitness:
                    best_fitness = fitnesses[i]
                    best_mask = sources[i].copy()

            # ── Onlooker bees phase ──
            fit_sum = sum(fitnesses) + 1e-12
            probs = [f / fit_sum for f in fitnesses]
            for _ in range(n_food):
                idx = np.random.choice(n_food, p=probs)
                neighbour = self._neighbour(sources[idx], n_features)
                f = self._fitness(neighbour.astype(bool), X, y)
                if f > fitnesses[idx]:
                    sources[idx] = neighbour
                    fitnesses[idx] = f
                    trials[idx] = 0
                if f > best_fitness:
                    best_fitness = f
                    best_mask = neighbour.copy()

            # ── Scout bees phase ──
            for i in range(n_food):
                if trials[i] >= self.limit:
                    sources[i] = np.random.randint(2, size=n_features)
                    if not np.any(sources[i]):
                        sources[i][np.random.randint(n_features)] = 1
                    fitnesses[i] = self._fitness(sources[i].astype(bool), X, y)
                    trials[i] = 0

        if best_mask is None or not np.any(best_mask):
            return np.ones(n_features, dtype=bool)
        return best_mask.astype(bool)
