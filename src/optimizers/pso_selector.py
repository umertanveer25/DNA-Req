# src/optimizers/pso_selector.py
"""Particle Swarm Optimization (PSO) based feature selector.

This is a lightweight PSO implementation that searches for a binary mask
over the feature space. It uses a simple velocity update and a fitness
function identical to the GA (linear SVC + 3‑fold macro‑F1).
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


class PSOFeatureSelector:
    def __init__(self, swarm_size=30, iterations=10, inertia=0.7, cognitive=1.5, social=1.5, mutation_rate=0.05, random_state=42):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.fitness_model = SVC(kernel="linear", C=1, random_state=self.random_state)

    def _fitness(self, mask, X, y):
        if not np.any(mask):
            return 0.0
        X_subset = X[:, mask]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.fitness_model, X_subset, y, cv=skf, scoring="f1_macro")
        return np.mean(scores)

    def select_features(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        # Initialise particles (positions) as binary masks
        particles = [np.random.randint(2, size=n_features) for _ in range(self.swarm_size)]
        velocities = [np.random.rand(n_features) for _ in range(self.swarm_size)]
        personal_best = particles.copy()
        personal_best_score = [self._fitness(p, X, y) for p in particles]
        # Global best
        gbest_idx = int(np.argmax(personal_best_score))
        gbest = personal_best[gbest_idx].copy()
        gbest_score = personal_best_score[gbest_idx]
        for _ in range(self.iterations):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (personal_best[i] - particles[i])
                    + self.social * r2 * (gbest - particles[i])
                )
                # Apply sigmoid to get probabilities and sample new position
                prob = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = (np.random.rand(n_features) < prob).astype(int)
                # Mutation (flip bits)
                mutation_mask = np.random.rand(n_features) < self.mutation_rate
                particles[i] = np.where(mutation_mask, 1 - particles[i], particles[i])
                # Evaluate fitness
                score = self._fitness(particles[i], X, y)
                if score > personal_best_score[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_score[i] = score
                if score > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score
        return gbest.astype(bool)
