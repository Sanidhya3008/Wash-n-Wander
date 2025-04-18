import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz
import sys
import os

# Import the TSP GA module
from tsp_ga import TSPGA, plot_fitness_evolution, plot_diversity_evolution, plot_best_path

# Import the fuzzy controller module
from fuzzy_washing_machine_controller import (
    create_fuzzy_system, compute_wash_time, get_membership_degrees,
    plot_membership_functions, generate_surface_data, 
    get_rules_table, get_linguistic_descriptions
)

# Set page config for the integrated app
st.set_page_config(
    page_title="Soft Computing Project", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
/* Base styles that work for both light and dark themes */
.main-header {
    font-size: 2.5rem;
    color: var(--text-color, #1E88E5);
    text-align: center;
    margin-bottom: 1rem;
    padding-top: 1rem;
}
.app-header {
    font-size: 2rem;
    color: var(--header-color, #0D47A1);
    text-align: center;
    margin-bottom: 1rem;
    padding-top: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: var(--header-color, #0D47A1);
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.info-text {
    background-color: var(--info-bg, #E3F2FD);
    color: var(--info-text, black);
    padding: 1rem;
    border-radius: 5px;
    border-left: 5px solid var(--info-border, #1976D2);
}
.result-box {
    background-color: var(--result-bg, #E8F5E9);
    color: var(--result-text, black);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid var(--result-border, #43A047);
    margin-top: 1rem;
    text-align: center;
    font-size: 1.2rem;
}
.warning-text {
    background-color: var(--warning-bg, #FFF3E0);
    color: var(--warning-text, black);
    padding: 0.8rem;
    border-radius: 5px;
    border-left: 5px solid var(--warning-border, #FF9800);
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
.app-container {
    background-color: var(--container-bg, #f8f9fa);
    color: var(--container-text, black);
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    border: 1px solid var(--container-border, #dee2e6);
}

/* Theme-specific color variables */
:root {
    --text-color: #1E88E5;
    --header-color: #0D47A1;
    --info-bg: #E3F2FD;
    --info-text: black;
    --info-border: #1976D2;
    --result-bg: #E8F5E9;
    --result-text: black;
    --result-border: #43A047;
    --warning-bg: #FFF3E0;
    --warning-text: black;
    --warning-border: #FF9800;
    --container-bg: #f8f9fa;
    --container-text: black;
    --container-border: #dee2e6;
}

/* Dark theme overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --text-color: #64B5F6;
        --header-color: #90CAF9;
        --info-bg: #0D47A1;
        --info-text: white;
        --info-border: #1976D2;
        --result-bg: #1B5E20;
        --result-text: white;
        --result-border: #43A047;
        --warning-bg: #E65100;
        --warning-text: white;
        --warning-border: #FF9800;
        --container-bg: #263238;
        --container-text: white;
        --container-border: #455A64;
    }
}

/* Additional Streamlit-specific dark mode detection */
[data-testid="stAppViewContainer"] {
    --text-color: #1E88E5;
    --header-color: #0D47A1;
    --info-bg: #E3F2FD;
    --info-text: black;
    --info-border: #1976D2;
    --result-bg: #E8F5E9;
    --result-text: black;
    --result-border: #43A047;
    --warning-bg: #FFF3E0;
    --warning-text: black;
    --warning-border: #FF9800;
    --container-bg: #f8f9fa;
    --container-text: black;
    --container-border: #dee2e6;
}

[data-testid="stAppViewContainer"].dark {
    --text-color: #64B5F6;
    --header-color: #90CAF9;
    --info-bg: #0D47A1;
    --info-text: white;
    --info-border: #1976D2;
    --result-bg: #1B5E20;
    --result-text: white;
    --result-border: #43A047;
    --warning-bg: #E65100;
    --warning-text: white;
    --warning-border: #FF9800;
    --container-bg: #263238;
    --container-text: white;
    --container-border: #455A64;
}
</style>

<script>
// Add dark class to Streamlit container if dark theme is detected
const streamlitDoc = window.parent.document;
const selectedTheme = localStorage.getItem('theme');
const streamlitContainer = streamlitDoc.querySelector('[data-testid="stAppViewContainer"]');

if (streamlitContainer) {
    if (selectedTheme === 'dark' || 
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches && selectedTheme !== 'light')) {
        streamlitContainer.classList.add('dark');
    }
}
</script>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">AI Algorithms Dashboard</h1>', unsafe_allow_html=True)

# App navigation
app_selection = st.sidebar.radio(
    "Select Application",
    ["Home", "Travelling Salesman Problem (GA)", "Fuzzy Washing Machine Controller"]
)

# Initialize session state variables if they don't exist
if 'tsp_running' not in st.session_state:
    st.session_state.tsp_running = False
if 'best_path' not in st.session_state:
    st.session_state.best_path = None
if 'best_distance' not in st.session_state:
    st.session_state.best_distance = None
if 'improvement' not in st.session_state:
    st.session_state.improvement = None

# Home page
if app_selection == "Home":
    st.markdown("""
    <div class="info-text">
    Welcome to the Soft Computing Algorithms Dashboard! This application demonstrates two different AI techniques:
    
    1. **Genetic Algorithm** - Solving the Travelling Salesman Problem
    2. **Fuzzy Logic** - Controlling a washing machine based on dirtiness and grease levels
    
    Select an application from the sidebar to get started!
                
    Developed By Asher, Sanidhya, Dhruv, Aditya Under Dr. Aloke Datta
    </div>
    """, unsafe_allow_html=True)
    
    # Display info about each application
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-container">
        <h3>🧬 Travelling Salesman Problem</h3>
        <p>The TSP is a classic problem in computer science: finding the shortest route that visits all cities exactly once and returns to the starting city.</p>
        <p>Our implementation uses a genetic algorithm with:</p>
        <ul>
            <li>Mixed initialization strategies</li>
            <li>Rank-based selection with elitism</li>
            <li>Specialized crossover operators</li>
            <li>Multiple mutation techniques</li>
            <li>2-opt local search optimization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="app-container">
        <h3>🧠 Fuzzy Washing Machine Controller</h3>
        <p>This application demonstrates a washing machine controller using Mamdani's fuzzy logic approach.</p>
        <p>The system determines appropriate wash time based on:</p>
        <ul>
            <li>Clothes' dirtiness levels</li>
            <li>Grease levels</li>
            <li>Fuzzy rule base with linguistic variables</li>
            <li>Membership functions and defuzzification</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Travelling Salesman Problem (GA) page
elif app_selection == "Travelling Salesman Problem (GA)":
    st.markdown('<h2 class="app-header">🧬 Travelling Salesman Problem Solver</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
    This application solves the classic Travelling Salesman Problem (TSP) using a genetic algorithm. 
    Adjust the parameters, generate a problem instance, and watch how the algorithm evolves to find an optimal solution!
    </div>
    """, unsafe_allow_html=True)
    
    # Create a sidebar for parameters
    st.sidebar.markdown('<p class="sub-header">TSP Algorithm Controls</p>', unsafe_allow_html=True)

    # Create tabs for better organization
    parameter_tab, advanced_tab, matrix_tab = st.sidebar.tabs(["Basic Parameters", "Advanced Settings", "Problem Setup"])

    # Basic parameters in first tab
    with parameter_tab:
        st.header("Problem Size")
        num_cities = st.slider(
            "Number of Cities", 
            min_value=5, 
            max_value=50, 
            value=15, 
            step=1,
            help="The number of cities the salesman needs to visit"
        )
        
        st.header("Evolution Parameters")
        num_chromosomes = st.slider(
            "Population Size", 
            min_value=10, 
            max_value=200, 
            value=30, 
            step=10,
            help="How many different solutions to maintain in each generation"
        )
        
        num_iterations = st.slider(
            "Number of Generations", 
            min_value=10, 
            max_value=200, 
            value=30, 
            step=10,
            help="How many evolutionary cycles to run"
        )

    # Advanced parameters in second tab
    with advanced_tab:
        st.header("Genetic Operators")
        
        elite_size = st.slider(
            "Elite Size", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1,
            help="Number of best solutions to keep unchanged in each generation"
        )
        
        crossover_rate = st.slider(
            "Crossover Rate", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.8, 
            step=0.05,
            help="Probability of two solutions exchanging genetic material"
        )
        
        mutation_rate = st.slider(
            "Mutation Rate", 
            min_value=0.01, 
            max_value=0.5, 
            value=0.2, 
            step=0.05,
            help="Probability of random changes in solutions"
        )
        
        # Optional local search
        use_local_search = st.checkbox(
            "Use Local Search", 
            value=True,
            help="Apply 2-opt local optimization to improve solutions"
        )
        
        local_search_frequency = st.slider(
            "Local Search Frequency", 
            min_value=1, 
            max_value=10, 
            value=5, 
            step=1,
            disabled=not use_local_search,
            help="How often to apply local search (in generations)"
        )
        
        # Random seed
        seed = st.number_input(
            "Random Seed", 
            min_value=0, 
            max_value=9999, 
            value=42, 
            step=1,
            help="Seed for random number generation (for reproducibility)"
        )
        
        # Button to generate a new random seed
        if st.button("Generate New Random Seed", help="Create a new random seed for different results"):
            seed = random.randint(0, 9999)
            st.success(f"Generated new seed: {seed}")

    # Problem setup in third tab
    with matrix_tab:
        st.header("City Distribution")
        
        upload_option = st.radio(
            "Distance Matrix Source",
            ("Generate Random Cities", "Upload Distance Matrix CSV"),
            help="Choose how to define distances between cities"
        )
        
        distance_matrix = None
        city_positions = None
        
        if upload_option == "Generate Random Cities":
            # City distribution options
            distribution = st.selectbox(
                "City Distribution Pattern",
                options=["Uniform Random", "Clustered", "Circle", "Grid"],
                help="Pattern for generating city positions"
            )
            
            # Add a button to regenerate the random matrix
            regenerate_matrix = st.button(
                "Generate New City Positions", 
                help="Create a new random problem instance"
            )
            
            # Set random seeds
            random.seed(seed)
            np.random.seed(seed)
            
            # Store the current matrix in session state if it doesn't exist or regenerate was clicked
            if 'distance_matrix' not in st.session_state or regenerate_matrix:
                # Use the current time as part of the seed to ensure uniqueness
                matrix_seed = seed + int(time.time()) % 10000
                random.seed(matrix_seed)
                np.random.seed(matrix_seed)
                
                # Generate city positions based on selected distribution
                if distribution == "Uniform Random":
                    city_positions = {i: (random.random(), random.random()) for i in range(num_cities)}
                elif distribution == "Clustered":
                    # Create 2-3 clusters
                    clusters = []
                    for _ in range(random.randint(2, 3)):
                        clusters.append((random.random(), random.random()))
                    
                    city_positions = {}
                    for i in range(num_cities):
                        # Select a random cluster
                        cluster = random.choice(clusters)
                        # Add a city near that cluster
                        city_positions[i] = (
                            max(0, min(1, cluster[0] + random.gauss(0, 0.15))),
                            max(0, min(1, cluster[1] + random.gauss(0, 0.15)))
                        )
                elif distribution == "Circle":
                    city_positions = {}
                    center = (0.5, 0.5)
                    radius = 0.4
                    for i in range(num_cities):
                        angle = 2 * np.pi * i / num_cities
                        x = center[0] + radius * np.cos(angle)
                        y = center[1] + radius * np.sin(angle)
                        city_positions[i] = (x, y)
                else:  # Grid
                    # Create a grid layout with some randomness
                    side = int(np.ceil(np.sqrt(num_cities)))
                    city_positions = {}
                    for i in range(num_cities):
                        row = i // side
                        col = i % side
                        # Add a small random offset to make it less rigid
                        x = (col + 0.5) / side + random.uniform(-0.03, 0.03)
                        y = (row + 0.5) / side + random.uniform(-0.03, 0.03)
                        city_positions[i] = (x, y)
                
                # Calculate distance matrix
                distance_matrix = np.zeros((num_cities, num_cities))
                for i in range(num_cities):
                    for j in range(num_cities):
                        if i != j:
                            x1, y1 = city_positions[i]
                            x2, y2 = city_positions[j]
                            distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Store in session state
                st.session_state['distance_matrix'] = distance_matrix
                st.session_state['city_positions'] = city_positions
                
                # Reset the random seed for the GA
                random.seed(seed)
                np.random.seed(seed)
                
                st.success("Generated new city positions and distance matrix!")
            else:
                # Use the stored matrix
                distance_matrix = st.session_state['distance_matrix']
                city_positions = st.session_state['city_positions']
        else:
            uploaded_file = st.file_uploader(
                "Upload Distance Matrix CSV", 
                type="csv",
                help="CSV file with distances between cities (must be square matrix)"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if df.shape[0] == df.shape[1]:  # Check if matrix is square
                        distance_matrix = df.values
                        num_cities = distance_matrix.shape[0]
                        # Generate random positions for visualization since we don't have actual positions
                        city_positions = {i: (random.random(), random.random()) for i in range(num_cities)}
                        
                        # Store in session state
                        st.session_state['distance_matrix'] = distance_matrix
                        st.session_state['city_positions'] = city_positions
                    else:
                        st.error("Uploaded matrix is not square. Please upload a valid distance matrix.")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

    # Main content area - use tabs for organization
    tsp_tab1, tsp_tab2, tsp_tab3 = st.tabs(["Evolution Visualization", "Data View", "How It Works"])

    with tsp_tab1:
        # Create two columns for visualization
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("🗺️ City Map and Best Route")
            
            # Plot area for best path
            path_fig_placeholder = st.empty()
            
            # Display initial city positions if available
            if city_positions is not None and not st.session_state.tsp_running:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Check that we have positions for all cities before trying to plot
                if all(i in city_positions for i in range(num_cities)):
                    cities_x = [city_positions[i][0] for i in range(num_cities)]
                    cities_y = [city_positions[i][1] for i in range(num_cities)]
                    ax.scatter(cities_x, cities_y, c='blue', s=100)
                    
                    # Add city labels
                    for i in range(num_cities):
                        ax.text(city_positions[i][0], city_positions[i][1], str(i), fontsize=12)
                        
                    ax.set_title("City Positions")
                    ax.axis('equal')
                    ax.grid(True)
                    path_fig_placeholder.pyplot(fig)
                else:
                    ax.text(0.5, 0.5, "Generate city positions to visualize", 
                            horizontalalignment='center', fontsize=12)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    path_fig_placeholder.pyplot(fig)
                plt.close(fig)
            
            # Area for displaying result details
            result_details = st.empty()
            
            # Show results if available
            if st.session_state.best_path is not None:
                result_details.markdown(f"""
                ### 📊 Solution Details
                - **Best Path**: {st.session_state.best_path}
                - **Total Distance**: {st.session_state.best_distance:.4f}
                - **Improvement**: {st.session_state.improvement:.2f}%
                """)
        
        with col2:
            st.subheader("📈 Evolution Progress")
            
            # Plot area for fitness evolution
            fitness_fig_placeholder = st.empty()
            
            # Plot area for diversity evolution
            diversity_fig_placeholder = st.empty()
            
            # Progress indicators
            progress_container = st.empty()
            status_text = st.empty()

    with tsp_tab2:
        # Distance matrix visualization
        st.subheader("Distance Matrix")
        st.write("This matrix shows the distance between each pair of cities.")
        
        if distance_matrix is not None:
            # Create a heatmap of the distance matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(distance_matrix, cmap='viridis')
            plt.colorbar(im, ax=ax, label='Distance')
            ax.set_title('Distance Matrix Heatmap')
            ax.set_xlabel('City Index')
            ax.set_ylabel('City Index')
            st.pyplot(fig)
            plt.close(fig)
            
            # Also show the raw data
            st.write("Raw distance data:")
            st.dataframe(pd.DataFrame(distance_matrix))
        else:
            st.info("Generate or upload a distance matrix to see the data.")

    with tsp_tab3:
        st.subheader("How to Use This Application")
        
        st.markdown("""
        ### Quick Start Guide
        
        1. **Set Parameters**: Use the sidebar to adjust the problem size and algorithm settings
        2. **Generate Cities**: Click "Generate New City Positions" to create a random problem
        3. **Run Algorithm**: Click the "Run Genetic Algorithm" button below
        4. **Analyze Results**: Watch the evolution progress and final solution quality
        
        ### Understanding the Parameters
        
        - **Number of Cities**: More cities = harder problem
        - **Population Size**: Larger populations explore more solutions but run slower
        - **Number of Generations**: More generations = more time to evolve better solutions
        - **Elite Size**: How many top solutions to preserve in each generation
        - **Crossover Rate**: How often solutions exchange information (0.8 = 80% chance)
        - **Mutation Rate**: How often random changes occur (0.2 = 20% chance)
        - **Local Search**: Uses 2-opt optimization to improve promising solutions
        
        ### Tips for Better Results
        
        - For difficult problems (many cities), use larger populations and more generations
        - If solutions converge too quickly, increase mutation rate or decrease elite size
        - Try different city distributions to see how they affect solution difficulty
        - Use the same random seed to compare different parameter settings on the same problem
        """)

    # Function to run the GA and update the UI
    def run_tsp_ga():
        if distance_matrix is None:
            st.error("Please set up city positions first!")
            return
        
        # Set state to running
        st.session_state.tsp_running = True
        
        # Create progress indicators
        progress_container = tsp_tab1.container()
        progress_bar = progress_container.progress(0)
        status_container = progress_container.empty()
        
        # Initialize the GA
        ga = TSPGA(
            distances=distance_matrix,
            pop_size=num_chromosomes,
            elite_size=elite_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_iterations=num_iterations
        )
        
        # Initial best solution
        initial_best = max(ga.population, key=lambda x: x.fitness)
        initial_distance = initial_best.get_distance()
        
        # Function to update UI during evolution
        def update_ui(iteration, best_chromosome, ga):
            # Update progress bar
            progress = min((iteration + 1) / num_iterations, 1.0)
            progress_bar.progress(progress)
            
            # Update status text
            current_distance = best_chromosome.get_distance()
            improvement = ((initial_distance / current_distance) - 1) * 100
            status_container.text(
                f"Generation {iteration + 1}/{num_iterations} - "
                f"Best Distance: {current_distance:.4f} - "
                f"Improvement: {improvement:.2f}% from initial"
            )
            
            # Apply local search if enabled and it's time
            if use_local_search and (iteration + 1) % local_search_frequency == 0:
                ga.apply_local_search()
            
            # Update plots every few iterations to avoid slowing down
            if (iteration + 1) % max(1, num_iterations//10) == 0 or iteration == num_iterations - 1 or iteration == 0:
                # Update fitness plot
                fitness_fig, fitness_ax = plt.subplots(figsize=(8, 4))
                plot_fitness_evolution(ga, fitness_fig, fitness_ax)
                fitness_fig_placeholder.pyplot(fitness_fig)
                plt.close(fitness_fig)
                
                # Update diversity plot
                diversity_fig, diversity_ax = plt.subplots(figsize=(8, 4))
                plot_diversity_evolution(ga, diversity_fig, diversity_ax)
                diversity_fig_placeholder.pyplot(diversity_fig)
                plt.close(diversity_fig)
                
                # Update path plot
                path_fig, path_ax = plt.subplots(figsize=(10, 6))
                plot_best_path(ga, city_positions, path_fig, path_ax)
                path_fig_placeholder.pyplot(path_fig)
                plt.close(path_fig)
                
                # Brief pause to allow UI to update
                time.sleep(0.1)
        
        # Run the GA with UI updates
        best_solution = ga.evolve(callback=update_ui)
        
        # Final UI update with solution details
        final_distance = best_solution.get_distance()
        improvement = ((initial_distance / final_distance) - 1) * 100
        
        # Store results in session state
        st.session_state.best_path = best_solution.path
        st.session_state.best_distance = final_distance
        st.session_state.improvement = improvement
        
        # Update result details
        result_details.markdown(f"""
        ### 📊 Solution Details
        - **Best Path**: {best_solution.path}
        - **Total Distance**: {final_distance:.4f}
        - **Initial Distance**: {initial_distance:.4f}
        - **Improvement**: {improvement:.2f}%
        """)
        
        # Final path visualization
        path_fig, path_ax = plt.subplots(figsize=(10, 6))
        plot_best_path(ga, city_positions, path_fig, path_ax)
        path_fig_placeholder.pyplot(path_fig)
        plt.close(path_fig)
        
        # Set state to not running
        st.session_state.tsp_running = False
        
        return ga, best_solution

    # Run button - centered and prominent
    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("🚀 Run Genetic Algorithm", 
                     use_container_width=True, 
                     disabled=st.session_state.tsp_running or distance_matrix is None):
            with st.spinner("Evolving solutions..."):
                ga, best_solution = run_tsp_ga()
                st.success("Algorithm completed! Explore the results above.")

# Fuzzy Washing Machine Controller page
elif app_selection == "Fuzzy Washing Machine Controller":
    st.markdown('<h2 class="app-header">🧠 Fuzzy Washing Machine Controller</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    This application demonstrates a washing machine controller using Mamdani's fuzzy logic approach.
    The system determines appropriate wash time based on clothes' dirtiness and grease levels.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for control panel
    st.sidebar.markdown('<p class="sub-header">Fuzzy Controller Settings</p>', unsafe_allow_html=True)
    
    # Input Methods in Sidebar
    input_method = st.sidebar.radio(
        "Input Method:",
        ["Sliders", "Text Fields"]
    )
    
    # Input area in sidebar
    st.sidebar.markdown('<p class="sub-header">Input Parameters</p>', unsafe_allow_html=True)
    
    # Based on input method choice
    if input_method == "Sliders":
        dirt_level = st.sidebar.slider(
            "Dirtiness Level", 
            min_value=0, 
            max_value=100, 
            value=50, 
            help="Set the level of dirt on clothes (0-100)"
        )
        
        grease_level = st.sidebar.slider(
            "Grease Level", 
            min_value=0, 
            max_value=50, 
            value=25,
            help="Set the level of grease on clothes (0-50)"
        )
    else:  # Text Fields
        dirt_col, grease_col = st.sidebar.columns(2)
        
        with dirt_col:
            dirt_level = st.number_input(
                "Dirtiness Level (0-100)", 
                min_value=0, 
                max_value=100, 
                value=50,
                step=1
            )
        
        with grease_col:
            grease_level = st.number_input(
                "Grease Level (0-50)", 
                min_value=0, 
                max_value=50, 
                value=25,
                step=1
            )
    
    # Calculate button in sidebar
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    calculate_button = st.sidebar.button("Calculate Wash Time", use_container_width=True)
    
    # Quick reference in sidebar
    with st.sidebar.expander("Quick Reference Guide", expanded=False):
        st.markdown("""
        ### Input Scales:
        - **Dirtiness**: 0-100 (VSD → VHD)
        - **Grease**: 0-50 (SG → HG)
        
        ### Output Scale:
        - **Wash Time**: 0-60 minutes (VST → VHT)
        
        ### Abbreviations:
        - VSD: Very Slightly Dirty
        - SD: Slightly Dirty
        - MD: Moderately Dirty
        - HD: Heavily Dirty
        - VHD: Very Heavily Dirty
        - SG: Slight Grease
        - MG: Moderate Grease
        - HG: Heavy Grease
        - VST: Very Short Time
        - ST: Short Time
        - MT: Medium Time
        - HT: High Time
        - VHT: Very High Time
        """)
    
    # Create the fuzzy system
    wash_ctrl, dirtiness, grease, wash_time = create_fuzzy_system()
    
    # Create tabs in main area
    fuzzy_tab1, fuzzy_tab2, fuzzy_tab3, fuzzy_tab4 = st.tabs([
        "📊 Results", 
        "📈 Membership Functions", 
        "🔍 Visualization", 
        "📋 Rules Table"
    ])
    
    with fuzzy_tab1:
        st.markdown('<p class="sub-header">Washing Machine Controller Results</p>', unsafe_allow_html=True)
        
        # Display current inputs
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            st.info(f"Current Dirtiness Level: **{dirt_level}**")
        with input_col2:
            st.info(f"Current Grease Level: **{grease_level}**")
        
        # Display result when calculate button is pressed
        if calculate_button:
            try:
                # Compute the result
                wash_result = compute_wash_time(wash_ctrl, dirt_level, grease_level)
                
                # Display result
                st.markdown(f"""
                <div class="result-box">
                <h2>Recommended Wash Time: {wash_result:.2f} minutes</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show membership levels
                st.markdown('<p class="sub-header">Membership Degrees for Current Inputs</p>', unsafe_allow_html=True)
                
                # Get membership degrees
                dirt_memberships, grease_memberships = get_membership_degrees(
                    dirtiness, grease, dirt_level, grease_level
                )
                
                # Visualization of active rules and membership
                active_rules_col1, active_rules_col2 = st.columns(2)
                
                with active_rules_col1:
                    st.write("**Dirtiness Membership Degrees:**")
                    dirt_df = pd.DataFrame(list(dirt_memberships.items()), 
                                          columns=['Dirtiness Level', 'Membership Degree'])
                    dirt_df = dirt_df.sort_values('Membership Degree', ascending=False)
                    
                    # Add color bars to the dataframe
                    dirt_df_styled = dirt_df.style.bar(subset=['Membership Degree'], 
                                                     color='#5c9dd5', 
                                                     vmin=0, 
                                                     vmax=1)
                    st.dataframe(dirt_df_styled, use_container_width=True)
                
                with active_rules_col2:
                    st.write("**Grease Membership Degrees:**")
                    grease_df = pd.DataFrame(list(grease_memberships.items()), 
                                            columns=['Grease Level', 'Membership Degree'])
                    grease_df = grease_df.sort_values('Membership Degree', ascending=False)
                    
                    # Add color bars to the dataframe
                    grease_df_styled = grease_df.style.bar(subset=['Membership Degree'], 
                                                     color='#5c9dd5', 
                                                     vmin=0, 
                                                     vmax=1)
                    st.dataframe(grease_df_styled, use_container_width=True)
                
                # Active rules visualization
                st.markdown('<p class="sub-header">Active Rules Visualization</p>', unsafe_allow_html=True)
                
                # Determine which rules are active
                active_dirt_terms = [term for term, degree in dirt_memberships.items() if degree > 0]
                active_grease_terms = [term for term, degree in grease_memberships.items() if degree > 0]
                
                # Get rules table
                rules_table = get_rules_table()
                
                # Create a visual representation of active rules
                highlighted_rules = rules_table.copy()
                
                # Apply styling for active cells
                def highlight_active_cells(s):
                    is_active = pd.Series(False, index=s.index)
                    for dirt_term in active_dirt_terms:
                        for grease_term in active_grease_terms:
                            if dirt_term == s.name and grease_term in highlighted_rules.columns:
                                is_active.loc[grease_term] = True
                    return ['background-color: #a8d08d' if v else '' for v in is_active]
                
                # Apply styling
                highlighted_rules = highlighted_rules.style.apply(highlight_active_cells, axis=1)
                highlighted_rules = highlighted_rules.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#0D47A1'), ('color', 'white')]},
                    {'selector': 'td', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
                ])
                
                st.dataframe(highlighted_rules, use_container_width=True)
                
                # Add explanation for active rules
                active_rule_explanations = []
                for dirt_term in active_dirt_terms:
                    for grease_term in active_grease_terms:
                        if dirt_term in rules_table.index and grease_term in rules_table.columns:
                            wash_term = rules_table.loc[dirt_term, grease_term]
                            dirt_degree = dirt_memberships[dirt_term]
                            grease_degree = grease_memberships[grease_term]
                            min_degree = min(dirt_degree, grease_degree)
                            
                            active_rule_explanations.append(
                                f"- Rule: IF dirtiness is {dirt_term} ({dirt_degree:.2f}) AND grease is {grease_term} "
                                f"({grease_degree:.2f}) THEN wash time is {wash_term} → Rule strength: {min_degree:.2f}"
                            )
                
                if active_rule_explanations:
                    st.write("**Active Rules:**")
                    for explanation in active_rule_explanations:
                        st.markdown(explanation)
                else:
                    st.warning("No rules are significantly active for the current input values.")
                
            except Exception as e:
                st.error(f"Error in computation: {e}")
        else:
            st.info("👈 Use the controls in the sidebar to set input values, then click 'Calculate Wash Time'")
    
    with fuzzy_tab2:
        st.markdown('<p class="sub-header">Membership Functions</p>', unsafe_allow_html=True)
        
        # Display membership functions
        st.markdown("### Dirtiness Membership Functions")
        fig_dirt = plot_membership_functions(dirtiness, "Dirtiness Membership Functions")
        st.pyplot(fig_dirt)
        
        st.markdown("### Grease Membership Functions")
        fig_grease = plot_membership_functions(grease, "Grease Membership Functions")
        st.pyplot(fig_grease)
        
        st.markdown("### Wash Time Membership Functions")
        fig_time = plot_membership_functions(wash_time, "Wash Time Membership Functions")
        st.pyplot(fig_time)
        
        # Add current values indicators if values are set
        if calculate_button:
            st.markdown('<p class="sub-header">Current Input Values on Membership Functions</p>', unsafe_allow_html=True)
            
            # Plot dirtiness with current value
            fig_current_dirt, ax = plt.subplots(figsize=(10, 6))
            for term in dirtiness.terms:
                ax.plot(dirtiness.universe, dirtiness.terms[term].mf, 
                        label=term, linewidth=2)
            
            # Add vertical line for current dirtiness value
            ax.axvline(x=dirt_level, color='red', linestyle='--', 
                      label=f'Current Value: {dirt_level}')
            
            # Add points where membership functions intersect with current value
            for term in dirtiness.terms:
                membership_degree = fuzz.interp_membership(
                    dirtiness.universe, dirtiness.terms[term].mf, dirt_level
                )
                if membership_degree > 0:
                    ax.plot(dirt_level, membership_degree, 'ro', markersize=8)
                    ax.text(dirt_level+2, membership_degree, 
                           f"{term}: {membership_degree:.2f}", 
                           verticalalignment='center')
            
            ax.set_title("Current Dirtiness Value on Membership Functions")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig_current_dirt)
            
            # Plot grease with current value
            fig_current_grease, ax = plt.subplots(figsize=(10, 6))
            for term in grease.terms:
                ax.plot(grease.universe, grease.terms[term].mf, 
                        label=term, linewidth=2)
            
            # Add vertical line for current grease value
            ax.axvline(x=grease_level, color='red', linestyle='--', 
                      label=f'Current Value: {grease_level}')
            
            # Add points where membership functions intersect with current value
            for term in grease.terms:
                membership_degree = fuzz.interp_membership(
                    grease.universe, grease.terms[term].mf, grease_level
                )
                if membership_degree > 0:
                    ax.plot(grease_level, membership_degree, 'ro', markersize=8)
                    ax.text(grease_level+1, membership_degree, 
                           f"{term}: {membership_degree:.2f}", 
                           verticalalignment='center')
            
            ax.set_title("Current Grease Value on Membership Functions")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig_current_grease)
    
    with fuzzy_tab3:
        st.markdown('<p class="sub-header">Visualization of Controller Behavior</p>', unsafe_allow_html=True)
        
        viz_options = st.multiselect(
            "Select Visualizations to Display:",
            ["3D Surface Plot", "Heatmap", "Contour Plot"],
            default=["3D Surface Plot", "Heatmap"]
        )
        
        # Generate surface data
        dirt_range, grease_range, results = generate_surface_data(wash_ctrl)
        X, Y = np.meshgrid(dirt_range, grease_range)
        
        # Display selected visualizations
        if "3D Surface Plot" in viz_options:
            st.markdown("### 3D Surface Plot")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y.T, results, cmap=cm.viridis, linewidth=0, antialiased=True)
            ax.set_xlabel('Dirtiness')
            ax.set_ylabel('Grease')
            ax.set_zlabel('Wash Time (minutes)')
            ax.set_title('Washing Machine Fuzzy Controller Response Surface')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Add marker for current point if calculate button was pressed
            if calculate_button:
                try:
                    wash_result = compute_wash_time(wash_ctrl, dirt_level, grease_level)
                    ax.scatter([dirt_level], [grease_level], [wash_result], 
                            color='red', s=100, marker='o', 
                            label=f'Current Input: ({dirt_level}, {grease_level}) → {wash_result:.1f} min')
                    ax.legend()
                except:
                    pass
                
            st.pyplot(fig)
        
        if "Heatmap" in viz_options:
            st.markdown("### Wash Time Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            heatmap = sns.heatmap(results, cmap='viridis', 
                                xticklabels=[f"{x:.0f}" for x in dirt_range[::4]],
                                yticklabels=[f"{y:.1f}" for y in grease_range[::4]],
                                ax=ax, annot=False)
            ax.set_xlabel('Dirtiness Level')
            ax.set_ylabel('Grease Level')
            ax.set_title('Wash Time Heatmap (minutes)')
            
            # Add marker for current point if calculate button was pressed
            if calculate_button:
                # Find the closest indices
                dirt_idx = np.abs(dirt_range - dirt_level).argmin()
                grease_idx = np.abs(grease_range - grease_level).argmin()
                ax.plot(dirt_idx, grease_idx, 'o', color='red', markersize=10)
                
            st.pyplot(fig)
        
        if "Contour Plot" in viz_options:
            st.markdown("### Contour Plot")
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = ax.contourf(X, Y.T, results, levels=20, cmap='viridis')
            ax.set_xlabel('Dirtiness Level')
            ax.set_ylabel('Grease Level')
            ax.set_title('Wash Time Contour Lines (minutes)')
            fig.colorbar(contour, ax=ax)
            
            # Add marker for current point if calculate button was pressed
            if calculate_button:
                ax.plot(dirt_level, grease_level, 'o', color='red', markersize=10, 
                       label=f'Current Input: ({dirt_level}, {grease_level})')
                ax.legend()
                
            st.pyplot(fig)
    
    with fuzzy_tab4:
        st.markdown('<p class="sub-header">Fuzzy Rules Table</p>', unsafe_allow_html=True)
        
        # Create rules table
        rules_table = get_rules_table()
        dirtiness_desc, grease_desc, time_desc = get_linguistic_descriptions()
        
        # Display the rules table with styling
        st.markdown("### Fuzzy Rule Base")
        st.markdown("The table shows wash time based on dirtiness (rows) and grease (columns):")
        
        # Display styled dataframe
        rules_styled = rules_table.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#0D47A1'), ('color', 'white')]},
            {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#E3F2FD')]},
            {'selector': 'td', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
        ])
        
        # Highlight cells if calculate button was pressed
        if calculate_button:
            # Get membership degrees
            dirt_memberships, grease_memberships = get_membership_degrees(
                dirtiness, grease, dirt_level, grease_level
            )
            
            # Determine which rules are active
            active_dirt_terms = [term for term, degree in dirt_memberships.items() if degree > 0]
            active_grease_terms = [term for term, degree in grease_memberships.items() if degree > 0]
            
            def highlight_active_cells(s):
                is_active = pd.Series(False, index=s.index)
                for dirt_term in active_dirt_terms:
                    for grease_term in active_grease_terms:
                        if dirt_term == s.name and grease_term in rules_table.columns:
                            is_active.loc[grease_term] = True
                return ['background-color: #a8d08d' if v else '' for v in is_active]
            
            # Apply styling
            rules_styled = rules_styled.apply(highlight_active_cells, axis=1)
        
        st.dataframe(rules_styled, use_container_width=True)
        
        # Add explanation for the table
        with st.expander("Rule Interpretation Guide", expanded=True):
            st.markdown("""
            ### How to interpret the rules table:
            1. Find the row matching the dirtiness level (VSD, SD, MD, HD, VHD)
            2. Find the column matching the grease level (SG, MG, HG)
            3. The cell at the intersection shows the recommended wash time (VST, ST, MT, HT, VHT)
            
            For example, if clothes are Moderately Dirty (MD) with Heavy Grease (HG), the wash time will be High Time (HT).
            """)
        
        # Display abbreviation descriptions
        st.markdown('<p class="sub-header">Linguistic Variables Description</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dirtiness Levels:")
            st.dataframe(dirtiness_desc.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0D47A1'), ('color', 'white')]}
            ]), use_container_width=True)
            
            st.markdown("#### Grease Levels:")
            st.dataframe(grease_desc.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0D47A1'), ('color', 'white')]}
            ]), use_container_width=True)
        
        with col2:
            st.markdown("#### Wash Time Levels:")
            st.dataframe(time_desc.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0D47A1'), ('color', 'white')]}
            ]), use_container_width=True)

# Footer information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: small;">
    Soft Computing Algorithms Dashboard | Genetic Algorithm and Fuzzy Logic Implementation
    
    Developed By Asher, Sanidhya, Dhruv, Aditya Under Dr. Aloke Datta
</div>
""", unsafe_allow_html=True)

# Main entrypoint
if __name__ == "__main__":
    pass