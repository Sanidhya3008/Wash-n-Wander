import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Import the fuzzy controller module
from fuzzy_washing_machine_controller import (
    create_fuzzy_system, compute_wash_time, get_membership_degrees,
    plot_membership_functions, generate_surface_data, 
    get_rules_table, get_linguistic_descriptions
)

def main():
    """Example usage of the fuzzy washing machine controller"""
    print("Fuzzy Washing Machine Controller Example")
    print("----------------------------------------")
    
    # Create the fuzzy system
    wash_ctrl, dirtiness, grease, wash_time = create_fuzzy_system()
    
    # Define test cases to demonstrate the system
    test_cases = [
        {"dirt": 10, "grease": 5, "description": "Very slightly dirty with slight grease"},
        {"dirt": 35, "grease": 15, "description": "Slightly to moderately dirty with moderate grease"},
        {"dirt": 60, "grease": 30, "description": "Moderately to heavily dirty with heavy grease"},
        {"dirt": 85, "grease": 45, "description": "Very heavily dirty with heavy grease"}
    ]
    
    # Process each test case
    print("\nTest Cases:")
    for i, case in enumerate(test_cases, 1):
        dirt_level = case["dirt"]
        grease_level = case["grease"]
        
        # Calculate wash time
        result = compute_wash_time(wash_ctrl, dirt_level, grease_level)
        
        # Print results
        print(f"\nCase {i}: {case['description']}")
        print(f"  Dirtiness: {dirt_level}, Grease: {grease_level}")
        print(f"  Recommended Wash Time: {result:.2f} minutes")
        
        # Get membership degrees
        dirt_memberships, grease_memberships = get_membership_degrees(
            dirtiness, grease, dirt_level, grease_level
        )
        
        # Print active memberships
        print("\n  Active Membership Degrees:")
        print("  Dirtiness:")
        for term, degree in dirt_memberships.items():
            if degree > 0.01:  # Only show significant memberships
                print(f"    {term}: {degree:.2f}")
        
        print("  Grease:")
        for term, degree in grease_memberships.items():
            if degree > 0.01:  # Only show significant memberships
                print(f"    {term}: {degree:.2f}")
    
    # Plot membership functions
    print("\nGenerating plots...")
    
    # Plot dirtiness membership functions
    fig_dirt = plot_membership_functions(dirtiness, "Dirtiness Membership Functions")
    plt.figure(fig_dirt.number)
    
    # Plot grease membership functions
    fig_grease = plot_membership_functions(grease, "Grease Membership Functions")
    plt.figure(fig_grease.number)
    
    # Plot wash time membership functions
    fig_time = plot_membership_functions(wash_time, "Wash Time Membership Functions")
    plt.figure(fig_time.number)
    
    # Generate and plot surface data
    dirt_range, grease_range, results = generate_surface_data(wash_ctrl)
    X, Y = np.meshgrid(dirt_range, grease_range)
    
    # 3D Surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y.T, results, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_xlabel('Dirtiness')
    ax.set_ylabel('Grease')
    ax.set_zlabel('Wash Time (minutes)')
    ax.set_title('Washing Machine Fuzzy Controller Response Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(results, cmap='viridis',
                        xticklabels=[f"{x:.0f}" for x in dirt_range[::4]],
                        yticklabels=[f"{y:.1f}" for y in grease_range[::4]],
                        ax=ax, annot=False)
    ax.set_xlabel('Dirtiness Level')
    ax.set_ylabel('Grease Level')
    ax.set_title('Wash Time Heatmap (minutes)')
    
    # Display the rules table
    rules_table = get_rules_table()
    print("\nFuzzy Rules Table:")
    print(rules_table)
    
    # Get linguistic descriptions
    dirtiness_desc, grease_desc, time_desc = get_linguistic_descriptions()
    
    print("\nLinguistic Variables:")
    print("\nDirtiness Levels:")
    print(dirtiness_desc)
    print("\nGrease Levels:")
    print(grease_desc)
    print("\nWash Time Levels:")
    print(time_desc)
    
    plt.show()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    import sys
    main()