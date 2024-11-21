import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict_species_distribution(test_data_path, model_path, output_csv):
    """
    Estima a distribuição de espécies usando MaxEnt e armazena as probabilidades em um arquivo CSV.
    
    :param test_data_path: Caminho para o arquivo CSV de entrada contendo as localizações geográficas e classes reais.
    :param model_path: Caminho para o modelo MaxEnt treinado.
    :param output_csv: Caminho para salvar o arquivo CSV de saída.
    """
    from joblib import load

    # Carregar dados de teste
    test_data = pd.read_csv(test_data_path)
    coordinates = test_data[['latitude', 'longitude']].values
    true_classes = test_data['true_class'].values

    # Carregar o modelo treinado (exemplo: MaxEnt implementado no Scikit-learn)
    model = load(model_path)

    # Estimar as probabilidades
    print("Estimando as probabilidades para cada classe...")
    probabilities = model.predict_proba(coordinates)

    # Organizar os resultados em um DataFrame
    rows = []
    for i, row in test_data.iterrows():
        record_id = row['record_id']
        true_class = row['true_class']
        prob_row = probabilities[i].tolist()
        rows.append([record_id, true_class] + prob_row)
    
    # Salvar os resultados no CSV
    columns = ['record_id', 'true_class'] + [f'class_{i}' for i in range(probabilities.shape[1])]
    output_df = pd.DataFrame(rows, columns=columns)
    output_df.to_csv(output_csv, index=False)
    print(f"Resultados salvos em {output_csv}")

# Configuração
test_data_path = "test_species_data.csv"  # Substitua pelo caminho do arquivo de teste
model_path = "maxent_model.joblib"  # Substitua pelo caminho do modelo treinado
output_csv = "species_distribution_predictions.csv"  # Nome do arquivo de saída

# Executar a predição
predict_species_distribution(test_data_path, model_path, output_csv)
