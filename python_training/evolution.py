import torch
import numpy as np
from typing import List
from model import RobotNet


class EvolutionaryAlgorithm:
    def __init__(self, population_size: int = 10, max_velocity: float = 300.0):
        self.population_size = population_size
        self.population: List[RobotNet] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        self.max_velocity = max_velocity
        
        # Initialize random population
        self.initialize_population()
    
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            model = RobotNet(hidden_size=32, max_velocity=self.max_velocity)
            self.population.append(model)
    
    def evolve(self, fitness_scores: List[float]) -> List[RobotNet]:
        self.fitness_scores = fitness_scores
        self.generation += 1
        
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        print(f"Generation {self.generation}")
        print(f"  Best fitness: {sorted_fitness[0]:.2f}")
        print(f"  Top 3 fitness: {sorted_fitness[:3]}")
        
        # Create new population using:
        # - Elitism: keep top 2 unchanged
        # - Selection + mutation: most offspring are mutated copies of good parents
        # - A small number of random newcomers for diversity
        new_population: List[RobotNet] = []
        
        # 1. Elites (keep best 2)
        elite_count = 2
        for i in range(elite_count):
            new_population.append(sorted_population[i].clone())
        
        # 2. Mutated offspring from selected parents
        # Use top half as candidate parents
        parent_pool_size = max(elite_count, self.population_size // 2)
        parent_pool = sorted_population[:parent_pool_size]
        
        # Reserve a small number of slots for random newcomers
        random_newcomers = 2
        max_mutated_offspring = self.population_size - elite_count - random_newcomers
        
        mutation_rate = 0.2
        mutation_strength = 0.3
        
        while len(new_population) < elite_count + max_mutated_offspring:
            # Tournament selection among parent_pool indices
            tournament_size = 3
            candidate_indices = np.random.randint(0, parent_pool_size, size=tournament_size)
            best_candidate_idx = int(np.min(candidate_indices))
            
            parent = parent_pool[best_candidate_idx]
            child = parent.clone()
            child.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            new_population.append(child)
        
        # 3. Random newcomers to maintain diversity
        while len(new_population) < self.population_size:
            new_model = RobotNet(hidden_size=32, max_velocity=self.max_velocity)
            new_population.append(new_model)
        
        self.population = new_population
        return self.population
    
    def get_best_model(self) -> RobotNet:
        if not self.fitness_scores:
            return self.population[0]
        
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]

