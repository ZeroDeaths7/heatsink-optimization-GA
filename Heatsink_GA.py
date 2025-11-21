import random
import math
import matplotlib.pyplot as plt

# ==========================================
# PART 1: THE SURROGATE PHYSICS MODEL
# ==========================================
# This function mimics the results you would get from ANSYS Fluent.
# It calculates a "Max Temperature" based on physical trade-offs.

def evaluate_heatsink_physics(individual):
    """
    Calculates the Max Temperature (T_max) for a given heatsink design.
    Lower is better.
    
    individual: A list of [fin_height, fin_thickness, fin_pitch, base_thickness]
    """
    h, t, p, b = individual
    
    # --- PHYSICS LOGIC ---
    
    # 1. Surface Area Benefit (Taller fins & more fins = better cooling)
    # We calculate "fin density" roughly as 1/pitch.
    # Baseline reference: Height 140mm, Pitch 2.2mm.
    area_score = (h / 140.0) * (2.2 / p)
    
    # 2. Airflow Choking Penalty (The "Sweet Spot" for Pitch)
    # If pitch is too small (< 1.5mm), air can't get in -> Huge Temp Spike.
    # If pitch is too big (> 3.5mm), not enough surface area -> Temp Increase.
    # Optimal is roughly 2.0mm - 2.4mm.
    optimal_pitch = 2.2
    if p < 0.8: # Manufacturing limit / Total blockage
        choke_penalty = 5000.0 # Failure
    else:
        # Parabolic penalty for moving away from optimal
        choke_penalty = 40.0 * ((p - optimal_pitch) ** 2)
        
    # Extra penalty for very tight spacing (exponential choking)
    if p < 1.5:
        choke_penalty += 50.0 * (1.5 - p)

    # 3. Conduction Benefit (Thicker base & fins spread heat better)
    # Thicker is better, but with diminishing returns.
    conduction_score = (t * 5.0) + (b * 1.5)
    
    # --- CALCULATE BASE TEMP (Kelvin) ---
    # Start with a "bad" baseline temp of 360K (87C)
    base_temp_k = 350.0
    
    # Apply factors:
    # - High Area Score reduces temp
    # - High Conduction Score reduces temp
    # - Choke Penalty adds temp
    
    cooling_factor = (area_score * 15.0) + conduction_score
    
    simulated_temp = base_temp_k - cooling_factor + choke_penalty
    
    # --- ADD REALISM (NOISE) ---
    # Real simulations have tiny variations.
    noise = random.uniform(-0.2, 0.2)
    final_temp = simulated_temp + noise
    
    # Clamp to realistic bounds (Ambient is ~300K)
    if final_temp < 305.0: final_temp = 305.0 + random.uniform(0, 1)
    
    return final_temp

# ==========================================
# PART 2: GENETIC ALGORITHM CONFIGURATION
# ==========================================

# Constraints / Design Space (Min, Max)
BOUNDS = [
    (130.0, 147.0),  # Fin Height (mm)
    (0.5, 2.0),      # Fin Thickness (mm)
    (1.2, 3.5),      # Fin Pitch (mm)
    (3.0, 10.0)      # Baseplate Thickness (mm)
]

POPULATION_SIZE = 50   # Number of designs per generation
GENERATIONS = 30       # How many times to evolve
MUTATION_RATE = 0.1    # Chance of a random change
ELITISM_COUNT = 2      # Keep the best 2 designs unchanged

def create_individual():
    """Creates a random heatsink design within bounds."""
    return [random.uniform(low, high) for low, high in BOUNDS]

def mutate(individual):
    """Randomly changes one gene of an individual."""
    if random.random() < MUTATION_RATE:
        gene_idx = random.randint(0, 3)
        low, high = BOUNDS[gene_idx]
        # Change the value by a small random amount (drift), but keep inside bounds
        change = random.uniform(-2.0, 2.0)
        new_val = individual[gene_idx] + change
        # Clamp to bounds
        new_val = max(low, min(high, new_val))
        individual[gene_idx] = new_val
    return individual

def crossover(parent1, parent2):
    """Mixes two parents to create a child."""
    # Single point crossover
    point = random.randint(1, 3)
    child = parent1[:point] + parent2[point:]
    return child

# ==========================================
# PART 3: RUNNING THE OPTIMIZATION
# ==========================================

def run_genetic_algorithm():
    print("------------------------------------------------------------")
    print("  STARTING GENETIC ALGORITHM FOR HEATSINK OPTIMIZATION")
    print("------------------------------------------------------------")
    print(f"  Population Size: {POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print("  Objective: Minimize Max Temperature (T_max)")
    print("------------------------------------------------------------\n")

    # 1. Initialize Population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    global_best_design = None
    global_best_temp = float('inf')
    
    # Store history for plotting
    history_best_temp = []
    history_generations = []

    for gen in range(1, GENERATIONS + 1):
        # 2. Evaluate Fitness (Run "Simulation" for everyone)
        # Store as tuple: (design, temperature)
        scored_population = []
        for ind in population:
            temp = evaluate_heatsink_physics(ind)
            scored_population.append((ind, temp))
        
        # Sort by Temperature (Lowest is best)
        scored_population.sort(key=lambda x: x[1])
        
        # Update Global Best
        best_of_gen = scored_population[0]
        if best_of_gen[1] < global_best_temp:
            global_best_temp = best_of_gen[1]
            global_best_design = best_of_gen[0]

        # Store data for plotting
        history_best_temp.append(best_of_gen[1])
        history_generations.append(gen)

        # Print Stats for this Generation
        print(f"Generation {gen:02d}: Best T_max = {best_of_gen[1]:.4f} K")
        
        # 3. Selection (Tournament Selection) & Breeding
        next_generation = []
        
        # Elitism: Carry over the absolute best ones directly
        for i in range(ELITISM_COUNT):
            next_generation.append(scored_population[i][0])
            
        # Fill the rest of the population
        while len(next_generation) < POPULATION_SIZE:
            # Select two parents randomly from top 50%
            top_half = scored_population[:POPULATION_SIZE//2]
            parent1 = random.choice(top_half)[0]
            parent2 = random.choice(top_half)[0]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            next_generation.append(child)
            
        population = next_generation

    # ==========================================
    # PART 4: FINAL RESULTS & PLOTTING
    # ==========================================
    print("\n------------------------------------------------------------")
    print("  OPTIMIZATION COMPLETE")
    print("------------------------------------------------------------")
    print(f"  Best Design Found:")
    print(f"   - Fin Height:        {global_best_design[0]:.2f} mm")
    print(f"   - Fin Thickness:     {global_best_design[1]:.2f} mm")
    print(f"   - Fin Pitch:         {global_best_design[2]:.2f} mm")
    print(f"   - Base Thickness:    {global_best_design[3]:.2f} mm")
    print(f"   ---------------------------------------")
    print(f"   - RESULTING T_MAX:   {global_best_temp:.4f} K")
    print("------------------------------------------------------------")
    
    # Plot the convergence graph
    plt.figure(figsize=(10, 6))
    plt.plot(history_generations, history_best_temp, marker='o', linestyle='-', color='b')
    plt.title('Genetic Algorithm Convergence: Min Temperature vs Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Temperature (K)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_genetic_algorithm()