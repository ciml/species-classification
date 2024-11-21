import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load

def train_maxent_model(train_data_path, model_path):
    """
    Treina um modelo MaxEnt para estimar a distribuição de espécies.
    
    :param train_data_path: Caminho para o arquivo CSV com os dados de treinamento.
    :param model_path: Caminho para salvar o modelo treinado.
    """
    # Carregar dados de treinamento
    train_data = pd.read_csv(train_data_path)
    
    # Separar variáveis preditoras (ambientais) e rótulos
    X = train_data.drop(columns=["record_id", "true_class"])
    y = train_data["true_class"]
    
    # Padronizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Treinar o modelo MaxEnt (usando Gradient Boosting como proxy)
    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)
    
    # Salvar modelo e scaler
    dump(model, model_path)
    dump(scaler, model_path.replace(".joblib", "_scaler.joblib"))
    print(f"Modelo salvo em {model_path}")

def predict_species_distribution(test_data_path, model_path, output_csv):
    """
    Estima a distribuição de espécies em dados de teste e salva as probabilidades em um arquivo CSV.
    
    :param test_data_path: Caminho para o arquivo CSV com os dados de teste.
    :param model_path: Caminho para o modelo MaxEnt treinado.
    :param output_csv: Caminho para salvar o arquivo CSV com as previsões.
    """
    # Carregar dados de teste
    test_data = pd.read_csv(test_data_path)
    record_ids = test_data["record_id"]
    true_classes = test_data["true_class"]
    X_test = test_data.drop(columns=["record_id", "true_class"])
    
    # Carregar modelo e scaler
    model = load(model_path)
    scaler = load(model_path.replace(".joblib", "_scaler.joblib"))
    
    # Padronizar dados de teste
    X_test_scaled = scaler.transform(X_test)
    
    # Estimar probabilidades
    print("Estimando probabilidades...")
    probabilities = model.predict_proba(X_test_scaled)
    
    # Organizar resultados
    rows = []
    for i, record_id in enumerate(record_ids):
        row = [record_id, true_classes[i]] + probabilities[i].tolist()
        rows.append(row)
    
    # Salvar no CSV
    columns = ["record_id", "true_class"] + [f"class_{i}" for i in range(probabilities.shape[1])]
    output_df = pd.DataFrame(rows, columns=columns)
    output_df.to_csv(output_csv, index=False)
    print(f"Resultados salvos em {output_csv}")

# Configuração
train_data_path = "train_species_data.csv"  # Caminho do arquivo de treinamento
test_data_path = "test_species_data.csv"  # Caminho do arquivo de teste
model_path = "maxent_model.joblib"  # Caminho para salvar o modelo treinado
output_csv = "species_distribution_predictions.csv"  # Caminho para salvar previsões

# Executar
print("Treinando modelo...")
train_maxent_model(train_data_path, model_path)

print("Realizando predições...")
predict_species_distribution(test_data_path, model_path, output_csv)
