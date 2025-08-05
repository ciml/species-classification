import pandas as pd
import numpy as np
import os

# Diretórios
data_dir = "dados_climaticos_ocorrencias"
output_dir = "resultados_bioclim"
os.makedirs(output_dir, exist_ok=True)

# Lista de classes
classes = [
    "anfibio", "ave-de-rapina", "cachorro", "capivara", "cobra",
    "cutia", "gamba", "lagarto", "macaco", "mico",
    "morcego", "preguica", "quati", "tamandua", "tartaruga"
]

# Grade climática com todos os pontos de previsão
grade = pd.read_csv("grade_climatica.csv")  # deve conter colunas bio1, bio2, ..., bio19

# Resultado geral
resultados = []

for classe in classes:
    print(f"Processando: {classe}")
    
    # Carrega as ocorrências da classe
    ocorrencias = pd.read_csv(f"{data_dir}/{classe}.csv")  # deve conter colunas bio1, bio2, ..., bio19
    
    # Seleciona apenas variáveis ambientais (supondo bio1-bio19)
    variaveis = [col for col in ocorrencias.columns if col.startswith("bio")]
    env_ocorrencias = ocorrencias[variaveis]
    
    # Define limites (envelope climático)
    limites_min = env_ocorrencias.min()
    limites_max = env_ocorrencias.max()
    
    # Aplica o envelope à grade
    adequado = np.ones(len(grade), dtype=bool)
    for var in variaveis:
        adequado &= (grade[var] >= limites_min[var]) & (grade[var] <= limites_max[var])
    
    # Adiciona coluna de adequação (1 = adequado, 0 = inadequado)
    predicao = pd.DataFrame({
        "classe": classe,
        "latitude": grade["latitude"],
        "longitude": grade["longitude"],
        "adequado": adequado.astype(int)
    })

    # Salva resultado individual
    predicao.to_csv(f"{output_dir}/bioclim_{classe}.csv", index=False)
    
    resultados.append(predicao)

# Junta tudo
todos_resultados = pd.concat(resultados, ignore_index=True)
todos_resultados.to_csv(f"{output_dir}/bioclim_todas_classes.csv", index=False)
print("✅ Bioclim finalizado.")
