import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def save_predictions_to_csv(model_path, dataset_paths, csv_output_paths, batch_size=32, image_size=(224, 224)):
    """
    Gera previsões e probabilidades para conjuntos de treinamento, validação e teste e salva os resultados em arquivos CSV.
    
    :param model_path: Caminho para o modelo treinado.
    :param dataset_paths: Dicionário com caminhos dos conjuntos de treinamento, validação e teste.
    :param csv_output_paths: Dicionário com caminhos para salvar os arquivos CSV de saída.
    :param batch_size: Tamanho do lote para processamento das imagens.
    :param image_size: Dimensões das imagens esperadas pelo modelo.
    """
    # Carregar o modelo treinado
    model = load_model(model_path)
    
    # Criar gerador de dados
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    for dataset_type, dataset_path in dataset_paths.items():
        print(f"Processando {dataset_type}...")
        
        # Configurar o gerador de dados
        data_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False  # Para garantir alinhamento entre arquivos e previsões
        )
        
        # Obter rótulos reais e nomes de arquivos
        file_names = data_generator.filenames
        true_classes = data_generator.classes
        class_indices = data_generator.class_indices
        class_labels = {v: k for k, v in class_indices.items()}  # Mapear índices para rótulos de classe
        
        # Gerar previsões
        print(f"Gerando previsões para {dataset_type}...")
        probabilities = model.predict(data_generator, verbose=1)
        
        # Organizar dados para salvar no CSV
        rows = []
        for i, file_name in enumerate(file_names):
            row = [file_name, class_labels[true_classes[i]]] + probabilities[i].tolist()
            rows.append(row)
        
        # Criar DataFrame para salvar no CSV
        columns = ["image", "true_class"] + [f"class_{i}" for i in range(probabilities.shape[1])]
        df = pd.DataFrame(rows, columns=columns)
        
        # Salvar no arquivo CSV
        csv_output_path = csv_output_paths[dataset_type]
        df.to_csv(csv_output_path, index=False)
        print(f"Previsões para {dataset_type} salvas em {csv_output_path}")

# Configurações
model_path = "path/to/your_model.h5"  # Substitua pelo caminho do seu modelo treinado
dataset_paths = {
    "train": "path/to/train_dataset",  # Substitua pelo caminho do conjunto de treinamento
    "validation": "path/to/validation_dataset",  # Substitua pelo caminho do conjunto de validação
    "test": "path/to/test_dataset",  # Substitua pelo caminho do conjunto de teste
}
csv_output_paths = {
    "train": "train_predictions.csv",  # Nome do arquivo CSV para treinamento
    "validation": "validation_predictions.csv",  # Nome do arquivo CSV para validação
    "test": "test_predictions.csv",  # Nome do arquivo CSV para teste
}

# Executar o script
save_predictions_to_csv(model_path, dataset_paths, csv_output_paths)
