library(dismo)
library(raster)

# Função para prever e salvar as probabilidades
predict_species_distribution <- function(test_data_path, model_path, output_csv) {
  # Carregar dados de teste
  test_data <- read.csv(test_data_path)
  
  # Carregar o modelo MaxEnt treinado
  model <- readRDS(model_path)
  
  # Criar raster de entrada com coordenadas geográficas
  coords <- test_data[, c("longitude", "latitude")]
  
  # Estimar probabilidades
  probabilities <- predict(model, coords, type = "raw")
  
  # Organizar os resultados em um DataFrame
  result <- data.frame(
    record_id = test_data$record_id,
    true_class = test_data$true_class,
    probabilities
  )
  
  # Renomear colunas para incluir as classes
  colnames(result)[3:ncol(result)] <- paste0("class_", seq_len(ncol(probabilities)) - 1)
  
  # Salvar no arquivo CSV
  write.csv(result, file = output_csv, row.names = FALSE)
  cat("Resultados salvos em", output_csv, "\n")
}

# Configuração
test_data_path <- "test_species_data.csv"  # Caminho para os dados de teste
model_path <- "maxent_model.rds"  # Caminho para o modelo MaxEnt treinado
output_csv <- "species_distribution_predictions.csv"  # Nome do arquivo de saída

# Executar predição
predict_species_distribution(test_data_path, model_path, output_csv)
