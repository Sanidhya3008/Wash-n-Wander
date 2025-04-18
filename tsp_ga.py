import numpy as np
from typing import List, Tuple, Callable, Optional
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import time

class Chromosome:
    """Represents a single solution (tour) in the TSP problem"""
    def __init__(self, path: List[int], distances: np.ndarray):
        self.path = path
        self.distances = distances
        self.fitness = 0  # Will be calculated when needed
        self._distance = None  # Cache for distance calculation
    
    def get_distance(self) -> float:
        """Calculate the total distance of the tour"""
        if self._distance is None:
            total_distance = 0
            for i in range(len(self.path) - 1):
                total_distance += self.distances[self.path[i], self.path[i+1]]
            # Add distance from last to first city to complete the tour
            total_distance += self.distances[self.path[-1], self.path[0]]
            self._distance = total_distance
        return self._distance
    
    def calculate_fitness(self) -> float:
        """Calculate the fitness (higher for shorter distances)"""
        distance = self.get_distance()
        # Simple inverse - shorter distances get higher fitness
        self.fitness = 1.0 / distance
        return self.fitness
    
    def __repr__(self):
        return f"Chromosome(path={self.path}, distance={self.get_distance():.4f})"

class TSPGA:
    """Simplified TSP Genetic Algorithm solver"""
    def __init__(self, 
                 distances: np.ndarray,
                 pop_size: int = 15,
                 elite_size: int = 2,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 max_iterations: int = 20):
        self.distances = distances
        self.num_cities = distances.shape[0]
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        
        # Initialize tracking variables
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.best_paths_history = []
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Calculate initial fitness for all chromosomes
        for chromosome in self.population:
            chromosome.calculate_fitness()
        
        # Track initial best distance for improvement calculation
        self.init_best_distance = min(chr.get_distance() for chr in self.population)
    
    def _initialize_population(self) -> List[Chromosome]:
        """Initialize the population with diverse strategies"""
        population = []
        
        # Add nearest neighbor solutions from different starting points
        for i in range(min(3, self.num_cities)):
            population.append(Chromosome(self._nearest_neighbor_path(i), self.distances))
        
        # Add a greedy solution
        try:
            population.append(Chromosome(self._greedy_path(), self.distances))
        except:
            # If greedy fails, add a random solution instead
            path = list(range(self.num_cities))
            random.shuffle(path)
            population.append(Chromosome(path, self.distances))
        
        # Fill the rest with random solutions
        while len(population) < self.pop_size:
            path = list(range(self.num_cities))
            random.shuffle(path)
            population.append(Chromosome(path, self.distances))
        
        return population
    
    def _nearest_neighbor_path(self, start_city: int) -> List[int]:
        """Generate a path using the nearest neighbor heuristic"""
        path = [start_city]
        unvisited = set(range(self.num_cities))
        unvisited.remove(start_city)
        
        current_city = start_city
        while unvisited:
            next_city = min(unvisited, key=lambda city: self.distances[current_city, city])
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            
        return path
    
    def _greedy_path(self) -> List[int]:
        """Generate a path using a greedy algorithm"""
        # Start with city 0
        path = [0]
        unvisited = set(range(1, self.num_cities))
        
        while unvisited:
            current_city = path[-1]
            # Find the closest unvisited city
            next_city = min(unvisited, key=lambda city: self.distances[current_city, city])
            path.append(next_city)
            unvisited.remove(next_city)
        
        return path
    
    def _selection(self) -> List[Chromosome]:
        """Select chromosomes for reproduction using tournament selection"""
        # Always keep the elite chromosomes
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        selected = sorted_population[:self.elite_size]
        
        # Tournament selection for the rest
        tournament_size = 3
        while len(selected) < self.pop_size:
            # Select random individuals for the tournament
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            # Select the best from the tournament
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Ordered crossover (OX) - preserves order and position of a subset of cities"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Select start and end positions for crossover
        size = len(parent1.path)
        start, end = sorted(random.sample(range(size), 2))
        
        # Function to create child using OX
        def create_child(p1, p2):
            # Copy the selected segment from parent1
            child = [-1] * size
            for i in range(start, end + 1):
                child[i] = p1.path[i]
            
            # Fill the remaining positions with cities from parent2 in order
            p2_idx = 0
            child_idx = 0
            
            while child_idx < size:
                if child_idx >= start and child_idx <= end:
                    child_idx = end + 1
                    continue
                
                # Find next city from parent2 that's not already in child
                while p2.path[p2_idx] in child:
                    p2_idx = (p2_idx + 1) % size
                
                child[child_idx] = p2.path[p2_idx]
                child_idx += 1
                p2_idx = (p2_idx + 1) % size
            
            return Chromosome(child, self.distances)
        
        # Create both children
        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        
        # Calculate fitness for new children
        child1.calculate_fitness()
        child2.calculate_fitness()
        
        return child1, child2
    
    def _mutation(self, chromosome: Chromosome) -> Chromosome:
        """Apply mutation operators with increasing probability if stuck"""
        if random.random() > self.mutation_rate:
            return chromosome
        
        # Copy the path to avoid modifying the original
        path = chromosome.path.copy()
        
        # Choose a mutation strategy
        mutation_type = random.choice(["swap", "inversion", "insertion"])
        
        if mutation_type == "swap":
            # Swap two random cities
            i, j = random.sample(range(self.num_cities), 2)
            path[i], path[j] = path[j], path[i]
            
        elif mutation_type == "inversion":
            # Invert a subsequence
            i, j = sorted(random.sample(range(self.num_cities), 2))
            path[i:j+1] = reversed(path[i:j+1])
            
        else:  # insertion
            # Take a city and insert it at a new position
            i = random.randrange(self.num_cities)
            j = random.randrange(self.num_cities)
            if i != j:
                city = path.pop(i)
                path.insert(j, city)
        
        # Create and return new chromosome
        mutated = Chromosome(path, self.distances)
        mutated.calculate_fitness()
        return mutated
    
    def _two_opt_local_search(self, chromosome: Chromosome, max_iterations=20) -> Chromosome:
        """2-opt local search to improve a solution"""
        best_path = chromosome.path.copy()
        best_distance = chromosome.get_distance()
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try all possible 2-opt swaps
            for i in range(1, self.num_cities - 1):
                if improved:
                    break
                    
                for j in range(i + 1, self.num_cities):
                    # Skip adjacent edges
                    if j - i == 1:
                        continue
                    
                    # Try 2-opt swap: reverse the segment between i and j
                    new_path = best_path.copy()
                    new_path[i:j+1] = reversed(new_path[i:j+1])
                    
                    # Check if this improves the solution
                    new_chromosome = Chromosome(new_path, self.distances)
                    new_distance = new_chromosome.get_distance()
                    
                    if new_distance < best_distance:
                        best_path = new_path
                        best_distance = new_distance
                        improved = True
                        break
        
        return Chromosome(best_path, self.distances)
    
    def apply_local_search(self):
        """Apply 2-opt local search to the best solution"""
        # Find the best chromosome
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_idx = 0
        
        # Apply local search
        improved = self._two_opt_local_search(self.population[best_idx])
        improved.calculate_fitness()
        
        # Replace if better
        if improved.fitness > self.population[best_idx].fitness:
            self.population[best_idx] = improved
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity as average path difference"""
        if len(self.population) <= 1:
            return 0.0
            
        total_difference = 0
        pairs_compared = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Count positions where cities differ
                diff = sum(1 for k in range(self.num_cities) 
                          if self.population[i].path[k] != self.population[j].path[k])
                total_difference += diff
                pairs_compared += 1
        
        # Normalize by path length and number of comparisons
        return total_difference / (pairs_compared * self.num_cities) if pairs_compared > 0 else 0
    
    def _inject_diversity(self):
        """Add diversity when population converges by replacing worst individuals"""
        # Sort population by fitness (ascending - worst first)
        self.population.sort(key=lambda x: x.fitness)
        
        # Replace bottom 10% (at least 1) with new random individuals
        num_to_replace = max(1, self.pop_size // 10)
        
        for i in range(num_to_replace):
            # Create a new random path
            new_path = list(range(self.num_cities))
            random.shuffle(new_path)
            new_chromosome = Chromosome(new_path, self.distances)
            new_chromosome.calculate_fitness()
            
            # Replace one of the worst individuals
            self.population[i] = new_chromosome
    
    def evolve(self, callback=None):
        """Run the evolutionary process for the specified number of iterations"""
        # Calculate initial fitness for all chromosomes if not already done
        for chromosome in self.population:
            if chromosome.fitness == 0:
                chromosome.calculate_fitness()
        
        # Record initial state
        best_chromosome = max(self.population, key=lambda x: x.fitness)
        self.best_fitness_history.append(best_chromosome.fitness)
        avg_fitness = sum(chr.fitness for chr in self.population) / len(self.population)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(self._calculate_diversity())
        self.best_paths_history.append(best_chromosome.path.copy())
        
        # Call the callback for initial state
        if callback:
            callback(0, best_chromosome, self)
        
        # Main evolution loop
        stagnation_counter = 0
        previous_best_distance = best_chromosome.get_distance()
        
        for iteration in range(self.max_iterations):
            # Selection
            selected = self._selection()
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism - directly copy the best chromosomes
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            new_population.extend(sorted_pop[:self.elite_size])
            
            # Crossover and mutation for the rest
            while len(new_population) < self.pop_size:
                # Select two parents
                parent1, parent2 = random.sample(selected, 2)
                
                # Create offspring through crossover and mutation
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                # Add to new population
                new_population.append(child1)
                new_population.append(child2)
                
                # Trim if needed
                if len(new_population) > self.pop_size:
                    new_population = new_population[:self.pop_size]
            
            # Update population
            self.population = new_population
            
            # Apply local search occasionally
            if (iteration + 1) % 5 == 0:  # Every 5 iterations
                self.apply_local_search()
            
            # Check diversity and inject if needed
            diversity = self._calculate_diversity()
            if diversity < 0.1:  # Very low diversity threshold
                self._inject_diversity()
            
            # Record statistics
            best_chromosome = max(self.population, key=lambda x: x.fitness)
            self.best_fitness_history.append(best_chromosome.fitness)
            avg_fitness = sum(chr.fitness for chr in self.population) / len(self.population)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            self.best_paths_history.append(best_chromosome.path.copy())
            
            # Check for stagnation
            current_best_distance = best_chromosome.get_distance()
            if abs(current_best_distance - previous_best_distance) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                previous_best_distance = current_best_distance
            
            # If stagnating for too long, increase mutation rate temporarily
            if stagnation_counter >= 3:
                old_rate = self.mutation_rate
                self.mutation_rate = min(self.mutation_rate * 1.5, 0.5)  # Increase but cap at 0.5
                self._inject_diversity()  # Force diversity
                self.mutation_rate = old_rate  # Restore original rate
                stagnation_counter = 0
            
            # Optional callback for UI updating
            if callback:
                callback(iteration + 1, best_chromosome, self)
        
        # Return the best solution found
        return max(self.population, key=lambda x: x.fitness)


# These visualization functions remain mostly the same as your original implementation
def plot_fitness_evolution(ga: TSPGA, fig=None, ax=None):
    """Plot the evolution of fitness over generations"""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = list(range(len(ga.best_fitness_history)))
    
    ax.plot(generations, ga.best_fitness_history, 'b-', label='Best Fitness')
    ax.plot(generations, ga.avg_fitness_history, 'r-', label='Average Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (1/Distance)')
    ax.set_title('Fitness Evolution')
    ax.legend()
    ax.grid(True)
    
    return fig, ax

def plot_diversity_evolution(ga: TSPGA, fig=None, ax=None):
    """Plot the evolution of population diversity over generations"""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = list(range(len(ga.diversity_history)))
    ax.plot(generations, ga.diversity_history, 'g-')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.set_title('Population Diversity Evolution')
    ax.grid(True)
    
    return fig, ax

def plot_best_path(ga: TSPGA, city_positions=None, fig=None, ax=None):
    """Plot the best path found by the GA"""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    best_chromosome = max(ga.population, key=lambda x: x.fitness)
    best_path = best_chromosome.path
    
    # If city positions are not provided, generate random positions
    if city_positions is None:
        city_positions = {i: (random.random(), random.random()) for i in range(ga.num_cities)}
    
    # Plot cities
    cities_x = [city_positions[i][0] for i in range(ga.num_cities)]
    cities_y = [city_positions[i][1] for i in range(ga.num_cities)]
    ax.scatter(cities_x, cities_y, c='blue', s=100)
    
    # Add city labels
    for i in range(ga.num_cities):
        ax.text(city_positions[i][0], city_positions[i][1], str(i), fontsize=12)
    
    # Plot path
    path_x = [city_positions[best_path[i]][0] for i in range(ga.num_cities)]
    path_y = [city_positions[best_path[i]][1] for i in range(ga.num_cities)]
    # Close the loop
    path_x.append(city_positions[best_path[0]][0])
    path_y.append(city_positions[best_path[0]][1])
    
    ax.plot(path_x, path_y, 'r-', linewidth=2)
    
    # Highlight starting city
    ax.scatter([city_positions[best_path[0]][0]], [city_positions[best_path[0]][1]], 
               c='green', s=200, zorder=5)
    
    ax.set_title(f'Best Path - Distance: {best_chromosome.get_distance():.2f}')
    ax.axis('equal')
    ax.grid(True)
    
    return fig, ax