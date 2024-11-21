import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random

# Função para carregar dados
def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e separa em ID, rótulos reais e previsões de ambos os modelos.
    """
    data = pd.read_csv(file_path)
    ids = data.iloc[:, 0].values  # ID dos registros
    y_true = data.iloc[:, 1].values  # Saídas reais
    model_1_probs = data.iloc[:, 2:17].values  # Probabilidades do Modelo 1
    model_2_probs = data.iloc[:, 17:32].values  # Probabilidades do Modelo 2
    return ids, y_true, model_1_probs, model_2_probs

# Função de avaliação
def evaluate(y_true, combined_probs):
    """
    Avalia a precisão combinando as previsões e calculando a acurácia.
    """
    y_pred = np.argmax(combined_probs, axis=1)
    return accuracy_score(y_true, y_pred)

# Combinação das probabilidades dos modelos
def combine_probabilities(model_1_probs, model_2_probs, weights):
    """
    Combina as probabilidades dos dois modelos usando pesos específicos para cada classe.
    """
    return weights * model_1_probs + (1 - weights) * model_2_probs

# Função para criar a população inicial
def create_population(size, num_classes):
    """
    Cria uma população inicial de pesos aleatórios para cada classe.
    """
    return np.random.rand(size, num_classes)

# Seleção dos melhores indivíduos
def select(population, scores, num_parents):
    """
    Seleciona os melhores indivíduos da população com base na pontuação.
    """
    selected_indices = np.argsort(scores)[-num_parents:]
    return population[selected_indices]

# Função de cruzamento
def crossover(parents, offspring_size, num_classes):
    """
    Realiza o cruzamento entre os pais para gerar novos indivíduos.
    """
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(list(parents), 2)
        child = (parent1 + parent2) / 2
        offspring.append(child)
    return np.array(offspring)

# Função de mutação
def mutate(offspring, mutation_rate=0.1):
    """
    Aplica mutação nos indivíduos da população.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            mutation = np.random.normal(0, 0.1, offspring.shape[1])
            offspring[i] += mutation  # Pequena alteração
            offspring[i] = np.clip(offspring[i], 0, 1)  # Mantém os pesos entre 0 e 1
    return offspring

# Algoritmo genético principal
def genetic_algorithm(y_true, model_1_probs, model_2_probs, num_generations=100, population_size=20, num_parents=10, mutation_rate=0.1):
    """
    Executa o algoritmo genético para encontrar os melhores pesos por classe.
    """
    num_classes = model_1_probs.shape[1]
    population = create_population(population_size, num_classes)  # População inicial
    
    for generation in range(num_generations):
        # Avaliação da população
        scores = []
        for weights in population:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weights)
            accuracy = evaluate(y_true, combined_probs)
            scores.append(accuracy)
        
        # Seleção dos melhores
        scores = np.array(scores)
        parents = select(population, scores, num_parents)
        
        # Cruzamento e mutação para gerar nova população
        offspring_size = population_size - num_parents
        offspring = crossover(parents, offspring_size, num_classes)
        offspring = mutate(offspring, mutation_rate)
        
        # Atualiza a população
        population = np.concatenate((parents, offspring))
        
        # Melhor indivíduo desta geração
        best_score = max(scores)
        best_weights = population[np.argmax(scores)]
        print(f"Geração {generation + 1} - Melhor precisão: {best_score:.4f}")
    
    # Melhor conjunto de pesos encontrado
    best_weights = population[np.argmax(scores)]
    return best_weights

# Carregar os dados
file_path = "dados.csv"  # Substitua pelo caminho do seu arquivo
ids, y_true, model_1_probs, model_2_probs = load_data(file_path)

# Executar o algoritmo genético
melhores_pesos = genetic_algorithm(y_true, model_1_probs, model_2_probs)
print(f"Melhores pesos encontrados para cada classe: {melhores_pesos}")
