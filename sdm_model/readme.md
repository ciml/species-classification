This file guides the use of the code modelR_sdm.R

###########################################
#Rscript modelR_sdm.R -c <arquivo_de_treinamento.csv> -o <nome_do_diretorio_do_modelo> -lon <lon> -lat <lat> -m <nome_do_diretorio_do_modelo_treinado>
#
###########################################
#Instalation of Model-R.

install_user_dir <- "~/R"
.libPaths( c( .libPaths(), install_user_dir ) ) ##Directory where R packages are installed.

## List of packages to be installed.
packages <- c("rJava", "raster", "dplyr", "devtools", "maps", "maptools", "rgeos", "argparse", "remotes")
instpack <- packages[!packages %in% installed.packages()]
if (length(instpack) > 0) {
  install.packages(packages[!packages %in% installed.packages()], lib=install_user_dir)
}

pack <- c("modleR")
instp <- pack[!pack %in% installed.packages()]
if (length(instp) > 0) {
  remotes::install_github("Model-R/modleR", build = TRUE, lib=install_user_dir)
}

# Required Packages
packages <- c("rJava", "raster", "dplyr", "devtools", "maps", "maptools", "rgeos", "argparse", "remotes")

# Initialization of the Packages required to run the Model-R
library(rJava)
library(raster)
library(modleR)
library(dplyr)
library(maps)
library(maptools)
library(rgeos)
library(argparse)

## Geração do modelo de distribuição de Espécies.
if (args$csv){
  
  if (args$modelfile){
  
    ## Creating an object with species names
    training_set <- read.table(args$csv, header=TRUE, sep= ",")
    species <- unique(training_set$species) 

# Creating the Buffer for generating pseudo-absences

##########################################################################
    # Geração dos Modelos de todas as espécies
    for (i in 1:length(species)) {
      sp <- species[i]
      occs <- training_set[training_set$species == species, c("lon", "lat")] 
      pts1 <- occs
     
      ##Criação do Buffer Mediana para o parâmetro min_geog_dist.
      names(pts1) = c("lon", "lat")
      buffer_type = "median"
      if(buffer_type=="median"){
            dist.buf <- median(spDists(x = pts1, longlat = FALSE, segments = TRUE))
      }
      
      ##Configuração das Pseudo-Ausências.
      setup_sdmdata(species_name = sp,
                    occurrences = occs,
                    models_dir = args$outputmodel,
                    predictors = example_vars,
                    buffer_type = "median",
                    min_geog_dist = 0.3*dist.buf
                    write_buffer = TRUE,
                    clean_dupl = FALSE,
                    clean_nas = FALSE,
                    clean_uni = FALSE,
                    png_sdmdata = FALSE,
                    n_back = length(occs),
                    partition_type = "crossvalidation",
                    cv_partitions = 1,
                    cv_n = 1
      )
    }

# MaxEnt Execution
##Teste do modelo gerado para verificação da espécie de um registro adicionado.
if (args$model){
  if (args$lon & args$lat){
    
    species <- read.table("./args$model/species.csv", header=TRUE, sep= ",") ##Leitura do arquivo contendo a lista de espécies.
    
    ##Modelo Final para verificação da classe de um novo registro.
    for (i in 1:length(species)) {
      sp <- species[i]
      occ <- [args$lon, args$lat]
      sp_final <- final_model(species_name = sp, 
                              occurrences = occ,
                              algorithms = "maxnet",
                              models_dir = args$model,
                              which_models = c("raw_mean"),
                              overwrite = TRUE)
    
      names(score) <- colnames(accuracy) ##Criando tabela para armazenar resultados da predição.
      score[i] <- sp_final.accuracy[,i]
    }


# Normalize the scores (i.e., calculate the probabilities, sum=1)
   soma <- sum(score)
   
   for (i in 1:length(score)){
      score[i] = score[i]/soma
    }

## Ordenação dos resultados para apresentação.
    res1 <- sort(score[,2], decreasing = T)
    ##Vetor com os índices após a ordenação da probabilidade.
    res2 <- order(score[,2], decreasing = T)

# Ordering results for presentation

