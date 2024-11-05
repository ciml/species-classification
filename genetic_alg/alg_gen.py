import numpy as np
import pandas as pd
import random

# Leitura do arquivo CSV
def carregar_dados(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    # Supondo que o CSV tem 30 colunas: 15 para o modelo 1 e 15 para o modelo 2
    modelo1 = df.iloc[:, :15].values  # Probabilidades do modelo 1
    modelo2 = df.iloc[:, 15:].values  # Probabilidades do modelo 2
    return modelo1, modelo2

# Função de avaliação (fitness)
# O fitness será a acurácia de classificação após a combinação dos modelos
def calcular_fitness(populacao, modelo1, modelo2):
    fitness = []
    for individuo in populacao:
        w1, w2 = individuo  # Pesos dos dois modelos

        acertos = 0
        for i in range(len(modelo1)):
            # Combina as probabilidades dos dois modelos
            prob_comb = w1 * modelo1[i] + w2 * modelo2[i]
            predicao = np.argmax(prob_comb)  # A classe com maior probabilidade
            classe_real = np.argmax(modelo1[i])  # Assume que a classe real é a do modelo 1
            if predicao == classe_real:
                acertos += 1

        # Acurácia como fitness
        fitness.append(acertos / len(modelo1))
    return fitness

# Inicialização da população
def inicializar_populacao(tamanho_populacao):
    # Cada indivíduo tem dois parâmetros: w1 (peso do modelo 1) e w2 (peso do modelo 2)
    # Inicia com valores aleatórios entre 0 e 1 para w1 e w2
    return [[random.random(), random.random()] for _ in range(tamanho_populacao)]

# Normalizar os pesos para somarem 1 (evita que um modelo tenha peso zero)
def normalizar_pesos(individuo):
    w1, w2 = individuo
    total = w1 + w2
    return [w1 / total, w2 / total]

# Seleção por torneio
def selecao(populacao, fitness):
    indice_pais = []
    for _ in range(len(populacao) // 2):
        torneio = random.sample(range(len(populacao)), 3)
        melhor_individuo = max(torneio, key=lambda i: fitness[i])
        indice_pais.append(melhor_individuo)
    return indice_pais

# Cruzamento de dois indivíduos
def cruzamento(pai1, pai2):
    # Cruzamento de um ponto: combinamos os pesos de cada modelo
    ponto_cruzamento = random.randint(0, 1)  # Cruzamos os dois parâmetros (w1, w2)
    if ponto_cruzamento == 0:
        filho1 = [pai1[0], pai2[1]]
        filho2 = [pai2[0], pai1[1]]
    else:
        filho1 = [pai2[0], pai1[1]]
        filho2 = [pai1[0], pai2[1]]
    return filho1, filho2

# Mutação de um indivíduo
def mutacao(individuo, taxa_mutacao):
    if random.random() < taxa_mutacao:
        # Muta um dos pesos aleatoriamente
        indice_mutacao = random.randint(0, 1)
        individuo[indice_mutacao] = random.random()  # Atribui um valor aleatório
    return normalizar_pesos(individuo)

# Algoritmo Genético
def algoritmo_genetico(caminho_arquivo, tamanho_populacao=100, num_geracoes=50, taxa_mutacao=0.1):
    modelo1, modelo2 = carregar_dados(caminho_arquivo)

    populacao = inicializar_populacao(tamanho_populacao)

    for geracao in range(num_geracoes):
        fitness = calcular_fitness(populacao, modelo1, modelo2)

        # Seleção dos pais
        pais = selecao(populacao, fitness)

        # Criação dos filhos
        nova_populacao = []
        for i in range(0, len(pais), 2):
            pai1 = populacao[pais[i]]
            pai2 = populacao[pais[i + 1]]
            filho1, filho2 = cruzamento(pai1, pai2)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        # Substitui a população antiga pela nova população
        populacao = nova_populacao

        # Exibir a melhor solução da geração
        melhor_fitness = max(fitness)
        print(f"Geração {geracao + 1}/{num_geracoes} - Melhor Fitness: {melhor_fitness:.4f}")

    # Melhor indivíduo após as gerações
    melhor_individuo = populacao[np.argmax(fitness)]
    return melhor_individuo

# Caminho do arquivo CSV
caminho_arquivo = "seu_arquivo.csv"

# Executa o algoritmo genético
melhor_solucao = algoritmo_genetico(caminho_arquivo)
print(f"\nMelhor solução encontrada: {melhor_solucao}")
