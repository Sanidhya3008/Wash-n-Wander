import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm

def create_fuzzy_system():
    """Create the fuzzy logic controller system for washing machine"""
    
    # Antecedents and consequent with proper ranges
    dirtiness = ctrl.Antecedent(np.arange(0, 101, 1), 'dirtiness')
    grease = ctrl.Antecedent(np.arange(0, 51, 1), 'grease')
    wash_time = ctrl.Consequent(np.arange(0, 61, 1), 'wash_time')
    
    # Membership functions with triangular shapes for dirtiness
    dirtiness['VSD'] = fuzz.trimf(dirtiness.universe, [0, 0, 25])
    dirtiness['SD'] = fuzz.trimf(dirtiness.universe, [0, 25, 50])
    dirtiness['MD'] = fuzz.trimf(dirtiness.universe, [25, 50, 75])
    dirtiness['HD'] = fuzz.trimf(dirtiness.universe, [50, 75, 100])
    dirtiness['VHD'] = fuzz.trimf(dirtiness.universe, [75, 100, 100])
    
    # Membership functions with triangular shapes for grease
    grease['SG'] = fuzz.trimf(grease.universe, [0, 0, 25])
    grease['MG'] = fuzz.trimf(grease.universe, [0, 25, 50])
    grease['HG'] = fuzz.trimf(grease.universe, [25, 50, 50])
    
    # Membership functions with triangular shapes for wash time
    wash_time['VST'] = fuzz.trimf(wash_time.universe, [0, 0, 15])
    wash_time['ST'] = fuzz.trimf(wash_time.universe, [0, 15, 30])
    wash_time['MT'] = fuzz.trimf(wash_time.universe, [15, 30, 45])
    wash_time['HT'] = fuzz.trimf(wash_time.universe, [30, 45, 60])
    wash_time['VHT'] = fuzz.trimf(wash_time.universe, [45, 60, 60])
    
    # Rule base according to the provided table
    rules = [
        # SG column
        ctrl.Rule(dirtiness['VSD'] & grease['SG'], wash_time['VST']),
        ctrl.Rule(dirtiness['SD'] & grease['SG'], wash_time['VST']),
        ctrl.Rule(dirtiness['MD'] & grease['SG'], wash_time['ST']),
        ctrl.Rule(dirtiness['HD'] & grease['SG'], wash_time['MT']),
        ctrl.Rule(dirtiness['VHD'] & grease['SG'], wash_time['HT']),
        
        # MG column
        ctrl.Rule(dirtiness['VSD'] & grease['MG'], wash_time['VST']),
        ctrl.Rule(dirtiness['SD'] & grease['MG'], wash_time['ST']),
        ctrl.Rule(dirtiness['MD'] & grease['MG'], wash_time['MT']),
        ctrl.Rule(dirtiness['HD'] & grease['MG'], wash_time['HT']),
        ctrl.Rule(dirtiness['VHD'] & grease['MG'], wash_time['VHT']),
        
        # HG column
        ctrl.Rule(dirtiness['VSD'] & grease['HG'], wash_time['ST']),
        ctrl.Rule(dirtiness['SD'] & grease['HG'], wash_time['MT']),
        ctrl.Rule(dirtiness['MD'] & grease['HG'], wash_time['HT']),
        ctrl.Rule(dirtiness['HD'] & grease['HG'], wash_time['VHT']),
        ctrl.Rule(dirtiness['VHD'] & grease['HG'], wash_time['VHT'])
    ]
    
    # Create control system
    wash_ctrl = ctrl.ControlSystem(rules)
    
    return wash_ctrl, dirtiness, grease, wash_time

def compute_wash_time(ctrl_sys, dirt_level, grease_level):
    """Compute the wash time for given input values"""
    washing_simulation = ctrl.ControlSystemSimulation(ctrl_sys)
    washing_simulation.input['dirtiness'] = dirt_level
    washing_simulation.input['grease'] = grease_level
    washing_simulation.compute()
    return washing_simulation.output['wash_time']

def get_membership_degrees(dirtiness, grease, dirt_level, grease_level):
    """Get membership degrees for input values"""
    dirt_memberships = {term: float(fuzz.interp_membership(dirtiness.universe, dirtiness.terms[term].mf, dirt_level)) 
                       for term in dirtiness.terms}
    grease_memberships = {term: float(fuzz.interp_membership(grease.universe, grease.terms[term].mf, grease_level)) 
                         for term in grease.terms}
    return dirt_memberships, grease_memberships

def plot_membership_functions(var, title):
    """Plot membership functions with custom styling"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '-', '-', '-', '-']
    
    for i, term in enumerate(var.terms):
        term_idx = list(var.terms.keys()).index(term)
        ax.plot(var.universe, var.terms[term].mf, 
                label=term, 
                linewidth=2.5, 
                color=colors[term_idx], 
                linestyle=linestyles[term_idx])
    
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    
    # Add vertical lines at membership function peaks
    for term in var.terms:
        if term in var.terms:
            max_idx = np.argmax(var.terms[term].mf)
            max_x = var.universe[max_idx]
            max_val = var.terms[term].mf[max_idx]
            if max_val > 0:
                ax.axvline(x=max_x, ymax=max_val, color='gray', linestyle=':', alpha=0.5)
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    fig.tight_layout()
    
    return fig

def generate_surface_data(ctrl_sys):
    """Generate surface data for 3D visualization"""
    simulator = ctrl.ControlSystemSimulation(ctrl_sys)
    
    # Create grid of points
    dirt_range = np.arange(0, 101, 5)
    grease_range = np.arange(0, 51, 2.5)
    
    # Initialize output array
    results = np.zeros((len(dirt_range), len(grease_range)))
    
    # Calculate output for each input combination
    for i, dirt_val in enumerate(dirt_range):
        for j, grease_val in enumerate(grease_range):
            simulator.input['dirtiness'] = dirt_val
            simulator.input['grease'] = grease_val
            simulator.compute()
            results[i, j] = simulator.output['wash_time']
    
    return dirt_range, grease_range, results

def get_rules_table():
    """Get the rules table as a pandas DataFrame"""
    rules_table = pd.DataFrame(
        [
            ["VST", "VST", "ST"],
            ["VST", "ST", "MT"],
            ["ST", "MT", "HT"],
            ["MT", "HT", "VHT"],
            ["HT", "VHT", "VHT"]
        ],
        index=["VSD", "SD", "MD", "HD", "VHD"],
        columns=["SG", "MG", "HG"]
    )
    return rules_table

def get_linguistic_descriptions():
    """Get descriptions of linguistic variables"""
    dirtiness_desc = pd.DataFrame({
        "Abbreviation": ["VSD", "SD", "MD", "HD", "VHD"],
        "Description": ["Very Slightly Dirty", "Slightly Dirty", "Moderately Dirty", 
                       "Heavily Dirty", "Very Heavily Dirty"]
    })
    
    grease_desc = pd.DataFrame({
        "Abbreviation": ["SG", "MG", "HG"],
        "Description": ["Slight Grease", "Moderate Grease", "Heavy Grease"]
    })
    
    time_desc = pd.DataFrame({
        "Abbreviation": ["VST", "ST", "MT", "HT", "VHT"],
        "Description": ["Very Short Time", "Short Time", "Medium Time", 
                       "High Time", "Very High Time"]
    })
    
    return dirtiness_desc, grease_desc, time_desc