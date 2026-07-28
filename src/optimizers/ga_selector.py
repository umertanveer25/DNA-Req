# src/optimizers/ga_selector.py
"""Genetic Algorithm based feature selector for Phase 3 pipelines.

The implementation mirrors the GA used in the original Phase 2‑A script. It
optimises a boolean mask over the 434‑dimensional DNA+SBERT feature space.
The fitness function uses a fast linear SVC with 3‑fold CV and returns the
average macro‑F1 score.

Usage
-----
>>> mask = GAFeatureSelector().select_features(X, y)
>>> X_opt = X[:, mask]
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


class GAFeatureSelector:
    def __init__(self, population_size=30, generations=10, mutation_rate=0.05, random_state=42):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.fitness_model = SVC(kernel="linear", C=1, random_state=self.random_state)

    def _fitness(self, chromosome, X, y):
        mask = chromosome == 1
        if not np.any(mask):
            return 0.0
        X_subset = X[:, mask]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.fitness_model, X_subset, y, cv=skf, scoring="f1_macro")
        return np.mean(scores)

    def select_features(self, X, y):
        """Run the GA and return a boolean mask of selected columns.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Full feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target labels.
        """
        np.random.seed(self.random_state)
        num_features = X.shape[1]
        # Initialise random population (1 = keep, 0 = drop)
        population = [np.random.randint(2, size=num_features) for _ in range(self.population_size)]
        best_chromosome = None
        best_fitness = 0.0
        for gen in range(self.generations):
            # Evaluate fitness for each chromosome
            fitness_scores = [self._fitness(chrom, X, y) for chrom in population]
            # Track best chromosome
            max_idx = int(np.argmax(fitness_scores))
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()
            # Selection: tournament (size 2)
            new_population = []
            for _ in range(self.population_size):
                i, j = np.random.choice(self.population_size, 2, replace=False)
                parent1 = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
                i, j = np.random.choice(self.population_size, 2, replace=False)
                parent2 = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
                # Uniform crossover
                child = np.where(np.random.rand(num_features) > 0.5, parent1, parent2)
                # Mutation
                mutations = np.random.rand(num_features) < self.mutation_rate
                child[mutations] = 1 - child[mutations]
                new_population.append(child)
            # Elitism – keep the absolute best
            new_population[0] = best_chromosome.copy()
            population = new_population
        # Final mask – ensure at least one feature is kept
        if best_chromosome is None or not np.any(best_chromosome):
            # fallback: keep all features
            return np.ones(num_features, dtype=bool)
        return best_chromosome.astype(bool)
