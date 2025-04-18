import numpy as np
import matplotlib.pyplot as plt
import random
from tsp_ga import TSPGA, plot_fitness_evolution, plot_diversity_evolution, plot_best_path

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    num_cities = 15
    pop_size = 15
    elite_size = 2
    crossover_rate = 0.8
    mutation_rate = 0.2
    max_iterations = 20
    
    # Generate random city positions
    city_positions = {i: (random.random(), random.random()) for i in range(num_cities)}
    
    # Calculate distance matrix
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x1, y1 = city_positions[i]
                x2, y2 = city_positions[j]
                distances[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Create and run the GA
    ga = TSPGA(
        distances=distances,
        pop_size=pop_size,
        elite_size=elite_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        max_iterations=max_iterations
    )
    
    # Callback function to print progress
    def print_progress(iteration, best_chromosome, ga):
        print(f"Iteration {iteration}: Best distance = {best_chromosome.get_distance():.4f}")
        
        # Apply local search every 5 iterations
        if iteration > 0 and iteration % 5 == 0:
            before_distance = best_chromosome.get_distance()
            ga.apply_local_search()
            after_best = max(ga.population, key=lambda x: x.fitness)
            after_distance = after_best.get_distance()
            print(f"  Applied local search: {before_distance:.4f} -> {after_distance:.4f}")
    
    # Run the evolution with the callback
    best_solution = ga.evolve(callback=print_progress)
    
    # Print final solution
    print("\nFinal Solution:")
    print(f"Path: {best_solution.path}")
    print(f"Distance: {best_solution.get_distance():.4f}")
    
    # Calculate improvement
    improvement_percent = ((ga.init_best_distance / best_solution.get_distance()) - 1) * 100
    print(f"Improvement: {improvement_percent:.2f}%")
    
    # Create plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    plot_fitness_evolution(ga, fig, axs[0])
    plot_diversity_evolution(ga, fig, axs[1])
    plt.tight_layout()
    
    # Plot the best path
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_best_path(ga, city_positions, fig, ax)
    plt.show()

if __name__ == "__main__":
    main()