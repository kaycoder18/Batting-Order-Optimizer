from flask import Flask, render_template, request
import random
import numpy as np

app = Flask(__name__)

# Function to get player statistics from user input
def get_player_stats(form_data):
    players = []
    for i in range(11):
        obp = float(form_data[f"obp_{i+1}"])
        slg = float(form_data[f"slg_{i+1}"])
        players.append([obp, slg])
    return players

def calculate_outcome_probabilities(obp, slg):
    """
    Calculate the probability of each outcome (single, double, triple, home run, out)
    based on OBP and SLG.
    """
    try:
        # Probability of getting on base
        p_on_base = obp
        # Probability of getting out
        p_out = 1 - obp
        
        # Average total bases per at-bat is SLG
        # We assume the distribution of hits (single, double, triple, home run) based on SLG
        hit_distribution = {
            'Single': 0.7,  # 70% of hits are singles
            'Double': 0.2,  # 20% of hits are doubles
            'Triple': 0.05,  # 5% of hits are triples
            'HomeRun': 0.05  # 5% of hits are home runs
        }
        
        # Calculate the probability of each hit type based on OBP and SLG
        p_single = p_on_base * hit_distribution['Single']
        p_double = p_on_base * hit_distribution['Double']
        p_triple = p_on_base * hit_distribution['Triple']
        p_homerun = p_on_base * hit_distribution['HomeRun']
        p_out = p_out  # The probability of an out remains the same as the calculated value for outs

        # Ensure probabilities sum to 1 (normalize if necessary)
        total_probability = p_single + p_double + p_triple + p_homerun + p_out
        
        # If the total probability is 0, set a default probability distribution to avoid division by zero
        if total_probability == 0:
            total_probability = 1  # Avoid division by zero, but this is unlikely
        
        # Normalize probabilities to sum to 1
        return [p_single / total_probability, p_double / total_probability, p_triple / total_probability, p_homerun / total_probability, p_out / total_probability]
    
    except ValueError as e:
        # Catch ValueError if probabilities are invalid or the sum is not 1
        print(f"Error calculating outcome probabilities: {e}")
        
        # Provide default probabilities if an error occurs (could be adjusted as needed)
        print("Using default probabilities.")
        
        # Default fallback probabilities (all outcomes equally likely)
        return [0.2, 0.2, 0.2, 0.2, 0.2]  # This ensures that probabilities sum to 1

# Simulate the inning
def simulate_inning(players, batting_order):
    runs = 0
    outs = 0
    bases = [0, 0, 0]  # [1st, 2nd, 3rd]
    current_batter = 0
    while outs < 3:
        player_index = batting_order[current_batter]
        obp, slg = players[player_index]
        outcome_probs = calculate_outcome_probabilities(obp, slg)
        outcome = np.random.choice(['Single', 'Double', 'Triple', 'HomeRun', 'Out'], p=outcome_probs)
        
        if outcome == 'Single':
            runs += bases[2]
            bases = [1, bases[0], bases[1]]
        elif outcome == 'Double':
            runs += bases[1] + bases[2]
            bases = [0, 1, bases[0]]
        elif outcome == 'Triple':
            runs += bases[0] + bases[1] + bases[2]
            bases = [0, 0, 1]
        elif outcome == 'HomeRun':
            runs += 1 + bases[0] + bases[1] + bases[2]
            bases = [0, 0, 0]
        elif outcome == 'Out':
            outs += 1
        current_batter = (current_batter + 1) % 11
    return min(runs, 5)

# Evaluate the batting order
def evaluate_batting_order(players, batting_order):
    total_runs = 0
    for _ in range(5):  # Simulate 5 innings
        total_runs += simulate_inning(players, batting_order)
    return total_runs

# Genetic algorithm and evolution functions
def create_individual():
    return random.sample(range(11), 11)

def create_population():
    return [create_individual() for _ in range(100)]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 10)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child

def mutate(individual):
    if random.random() < 0.1:
        idx1, idx2 = random.sample(range(11), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def evolve_population(population, players):
    evaluated_population = [(evaluate_batting_order(players, individual), individual) for individual in population]
    evaluated_population.sort(reverse=True, key=lambda x: x[0])
    next_generation = [individual for _, individual in evaluated_population[:50]]
    while len(next_generation) < 100:
        parent1, parent2 = random.sample(next_generation, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_generation.append(child)
    return next_generation

def genetic_algorithm(players):
    population = create_population()
    for generation in range(50):
        population = evolve_population(population, players)
        best_individual = max(population, key=lambda ind: evaluate_batting_order(players, ind))
        print(f"Generation {generation + 1}: Best Runs = {evaluate_batting_order(players, best_individual)}")
    best_individual = max(population, key=lambda ind: evaluate_batting_order(players, ind))
    return best_individual, evaluate_batting_order(players, best_individual)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        players = get_player_stats(request.form)
        best_order, best_runs = genetic_algorithm(players)
        return render_template("index.html", best_order=best_order, best_runs=best_runs)
    return render_template("index.html", best_order=None, best_runs=None)

if __name__ == "__main__":
    app.run(debug=True)
