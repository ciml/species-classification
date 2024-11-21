import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random

# Function to load data
def load_data(file_path):
    """
    Loads the data from the CSV file and separates it into ID, actual labels and predictions from both models.
    """
    data = pd.read_csv(file_path)
    ids = data.iloc[:, 0].values  # Record IDs
    y_true = data.iloc[:, 1].values  # Real exits
    model_1_probs = data.iloc[:, 2:17].values  # ResNet Probabilities
    model_2_probs = data.iloc[:, 17:32].values  # MaxEnt Probabilities
    return ids, y_true, model_1_probs, model_2_probs

# Evaluation function
def evaluate(y_true, combined_probs):
    """
    Evaluates accuracy by combining predictions and calculating accuracy.
    """
    y_pred = np.argmax(combined_probs, axis=1)
    return accuracy_score(y_true, y_pred)

# Combining model probabilities
def combine_probabilities(model_1_probs, model_2_probs, weight):
    """
    Combines the probabilities of the two models using a weight.
    """
    return weight * model_1_probs + (1 - weight) * model_2_probs

# Function to create the initial population
def create_population(size):
    """
    Creates an initial population of random weights between 0 and 1.
    """
    return np.random.rand(size)

# Selection of the best individuals
def select(population, scores, num_parents):
    """
    Selects the best individuals from the population based on the score.
    """
    selected_indices = np.argsort(scores)[-num_parents:]
    return population[selected_indices]

# Crossover function
def crossover(parents, offspring_size):
    """
    It crosses between parents to generate new individuals.
    """
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(list(parents), 2)
        child = (parent1 + parent2) / 2
        offspring.append(child)
    return np.array(offspring)

# Mutation function
def mutate(offspring, mutation_rate=0.1):
    """
    Applies mutation to individuals in the population.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] += np.random.normal(0, 0.1)  # Pequena alteração
            offspring[i] = np.clip(offspring[i], 0, 1)  # Mantém o peso entre 0 e 1
    return offspring

# Main genetic algorithm
def genetic_algorithm(y_true, model_1_probs, model_2_probs, num_generations=10000, population_size=1000, num_parents=50, mutation_rate=0.1):
    """
    Runs the genetic algorithm to find the best combination weight.
    """
    # Creation of the initial population
    population = create_population(population_size)
    
    for generation in range(num_generations):
        # Population assessment
        scores = []
        for weight in population:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weight)
            accuracy = evaluate(y_true, combined_probs)
            scores.append(accuracy)
        
        # Selection of the best
        scores = np.array(scores)
        parents = select(population, scores, num_parents)
        
        # Crossing and mutation to generate new population
        offspring_size = population_size - num_parents
        offspring = crossover(parents, offspring_size)
        offspring = mutate(offspring, mutation_rate)
        
        # Update the population
        population = np.concatenate((parents, offspring))
        
        # Best individual of this generation
        best_score = max(scores)
        best_weight = population[np.argmax(scores)]
        print(f"Generation {generation + 1} - Best score: {best_score:.4f}, Best weight: {best_weight:.4f}")
    
    # Best weight found
    best_weight = population[np.argmax(scores)]
    return best_weight

# Load the data
file_path = "dados.csv"  
ids, y_true, model_1_probs, model_2_probs = load_data(file_path)

# Run the genetic algorithm
melhor_peso = genetic_algorithm(y_true, model_1_probs, model_2_probs)
print(f"Best weight found: {melhor_peso}")
