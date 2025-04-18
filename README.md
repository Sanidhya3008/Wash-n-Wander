# AI Algorithms Dashboard

A comprehensive implementation of two artificial intelligence approaches: Genetic Algorithms for solving the Travelling Salesman Problem and Fuzzy Logic for a washing machine controller. This project integrates both techniques into a single interactive dashboard.

## Project Overview

### 1. TSP Genetic Algorithm Solver

This component implements an enhanced genetic algorithm for solving the Travelling Salesman Problem with features like:

- Mixed initialization strategies (random, nearest neighbor, greedy)
- Rank-based selection with elitism
- Specialized crossover with repair mechanisms
- Multiple mutation operators (inversion, insertion, swap)
- 2-opt local search for optimization
- Diversity maintenance techniques
- Interactive visualization

### 2. Fuzzy Washing Machine Controller

This component demonstrates a washing machine controller using Mamdani's fuzzy logic approach:

- Determines appropriate wash time based on clothes' dirtiness and grease levels
- Implements fuzzy membership functions for input and output variables
- Uses a rule base with linguistic variables
- Provides visual representation of membership functions and rule activation
- Demonstrates defuzzification through center of gravity method
- Includes 3D visualization of the controller's behavior

## Project Structure

- `tsp_ga.py` - Core GA implementation for TSP
- `fuzzy_washing_machine_controller.py` - Core fuzzy logic implementation
- `example_usage.py` - Script to demonstrate TSP GA usage
- `fuzzy_example.py` - Script to demonstrate Fuzzy Logic Controller usage
- `integrated_app.py` - Main application that integrates both components
- `requirements.txt` - Project dependencies

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:

   ```bash
   git clone <your-repository-url>
   cd ai-algorithms-dashboard
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Integrated Dashboard

Run the integrated application to access both implementations in a single interface:

```bash
streamlit run integrated_app.py
```

This will launch a web interface where you can:

- Navigate between both applications
- Adjust parameters for each algorithm
- Visualize results in real-time
- Explore detailed visualizations and explanations

### Individual Example Scripts

#### TSP Genetic Algorithm

Run the TSP GA example script:

```bash
python example_usage.py
```

This will:

- Generate a random TSP problem with 15 cities
- Run the genetic algorithm for 20 iterations
- Print progress and apply local search every 5 iterations
- Display the final solution and improvement
- Save visualizations of the fitness evolution and best path

#### Fuzzy Washing Machine Controller

Run the Fuzzy Logic example script:

```bash
python fuzzy_example.py
```

This will:

- Create a fuzzy control system with membership functions
- Compute wash time for sample input values
- Display the membership functions and rule activation
- Visualize the controller's behavior across the input range

## Example Usage: TSP Genetic Algorithm

```python
import numpy as np
import random
from tsp_ga import TSPGA

# Set parameters
num_cities = 15
distances = np.random.rand(num_cities, num_cities)
# Make the distance matrix symmetric
distances = (distances + distances.T) / 2
np.fill_diagonal(distances, 0)

# Create and run the GA
ga = TSPGA(
    distances=distances,
    pop_size=15,
    elite_size=2,
    crossover_rate=0.8,
    mutation_rate=0.2,
    max_iterations=20
)

# Define a callback to track progress
def print_progress(iteration, best_chromosome, ga):
    print(f"Iteration {iteration}: Best distance = {best_chromosome.get_distance():.4f}")

# Run the evolution
best_solution = ga.evolve(callback=print_progress)

# Print results
print(f"Best path: {best_solution.path}")
print(f"Distance: {best_solution.get_distance():.4f}")
```

## Example Usage: Fuzzy Washing Machine Controller

```python
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_washing_machine_controller import create_fuzzy_system, compute_wash_time

# Create the fuzzy control system
wash_ctrl, dirtiness, grease, wash_time = create_fuzzy_system()

# Define input values
dirt_level = 60  # Moderately to heavily dirty
grease_level = 30  # Moderate to heavy grease

# Calculate wash time
result = compute_wash_time(wash_ctrl, dirt_level, grease_level)
print(f"For dirtiness level {dirt_level} and grease level {grease_level}, the wash time is {result:.2f} minutes")

# Plot the membership functions to visualize
plt.figure(figsize=(10, 6))
for term in dirtiness.terms:
    plt.plot(dirtiness.universe, dirtiness.terms[term].mf, label=term)
plt.title("Dirtiness Membership Functions")
plt.legend()
plt.grid(True)
plt.show()
```

## Deployment Options

### 1. Streamlit Community Cloud (Free)

The easiest option for deploying this application is via Streamlit Community Cloud:

1. Push your code to GitHub
2. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Create a new app and point it to your repository
4. Follow the deployment instructions

### 2. Heroku Deployment

1. Create a `Procfile` in your project root:

   ```bash
   web: streamlit run integrated_app.py --server.port $PORT
   ```

2. Create a `requirements.txt` file if not already present:

   ```bash
   streamlit>=1.10.0
   numpy>=1.20.0
   pandas>=1.3.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   scikit-fuzzy>=0.4.2
   ```

3. Deploy to Heroku:

   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### 3. Docker Deployment

1. Create a `Dockerfile`:

   ```Dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "integrated_app.py"]
   ```

2. Build and run the Docker container:

   ```bash
   docker build -t ai-algorithms-dashboard .
   docker run -p 8501:8501 ai-algorithms-dashboard
   ```

3. For cloud deployment, push to Docker Hub:

   ```bash
   docker build -t yourusername/ai-algorithms-dashboard .
   docker push yourusername/ai-algorithms-dashboard
   ```

## Customization

The modular design makes it easy to extend both components:

### TSP Genetic Algorithm

- Add new initialization strategies
- Implement different selection methods
- Create alternative crossover operators
- Design new mutation operators
- Enhance the local search

### Fuzzy Washing Machine Controller

- Modify membership functions
- Add new input/output variables
- Change the rule base
- Implement different defuzzification methods
- Create alternative visualization techniques

## Troubleshooting

### Common Issues

1. **Dependencies installation failure**: Make sure you have a C++ compiler installed for scikit-fuzzy
2. **Performance issues**: For large problems, reduce visualization frequency
3. **Memory errors**: Lower population size and number of generations
4. **Streamlit display issues**: Check browser compatibility and try a different one
