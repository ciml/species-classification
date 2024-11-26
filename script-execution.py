import subprocess
import sys
import os

def execute_command(command, step_name):
    """
    Executa um comando no terminal e verifica seu sucesso.
    :param command: Comando a ser executado.
    :param step_name: Nome do passo para exibição.
    """
    print(f"Executando: {step_name}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Sucesso: {step_name}\nSaída: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Erro durante {step_name}.\nSaída: {e.stdout}\nErro: {e.stderr}")
        sys.exit(1)

def main():
    # Passo 1: Organização do Dataset
    step1_command = "python3 class_id/assign_class_index.py -c id_registro-id_animal-id_tipo.csv images/*"
    execute_command(step1_command, "Passo 1 - Organização do Dataset")

    # Passo 2: Aumento de Dados
    step2_command = "python3 data-augmentation/augmentation.py"
    execute_command(step2_command, "Passo 2 - Aumento de Dados")
   
    # Passo 2-1: Separação de Dados
    step21_command = "python3 data-augmentation/partition.py"
    execute_command(step21_command, "Passo 2 - Separação de Dados")

    # Passo 3: Treinamento do Modelo de Classificação de Imagens
    step3_command = ("python3 cnn_model/resnet-50.py ")
    execute_command(step3_command, "Passo 3 - Treinamento do Modelo de Classificação de Imagens")

    # Passo 4: Treinamento do Modelo de Distribuição de Espécies
    step4_command = "Rscript sdm_model/modelR_sdm.R -c arquivo_de_treinamento.csv"
    execute_command(step4_command, "Passo 4 - Treinamento do Modelo de Distribuição de Espécies")

    # Passo 5: Execução do Algoritmo Genético
    step5_command = "python3 genetic_alg/alg_gen.py -c arquivo_de_treinamento.csv"
    execute_command(step5_command, "Passo 5 - Execução do Algoritmo Genético")

if __name__ == "__main__":
    main()
