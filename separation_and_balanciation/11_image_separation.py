import os
import shutil
import pandas as pd
import re

# Função para sanitizar nomes de pastas
def sanitizar_nome_pasta(nome):
    """Mantém apenas letras, números, espaço, hífen e underscore. Troca o resto por '_'"""
    return re.sub(r'[\\/:*?"<>|]', "_", nome)

# Caminhos
pasta_imagens = "D:/Users/anton/Desktop/sissgeo/imagens"

saida_treino = 'D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/train/'
saida_val = 'D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/validation/'
saida_teste = 'D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/test/'

# Criar pastas de saída se não existirem
os.makedirs(saida_treino, exist_ok=True)
os.makedirs(saida_val, exist_ok=True)
os.makedirs(saida_teste, exist_ok=True)

def carregar_csv(caminho):
    df = pd.read_csv(caminho)
    df["Animal: Tipo"] = df["Animal: Tipo"].apply(sanitizar_nome_pasta)
    return df

# CSVs
treino_df = carregar_csv(os.path.join(saida_treino,"registros_treino.csv"))
val_df = carregar_csv(os.path.join(saida_val,"registros_validacao.csv"))
teste_df = carregar_csv(os.path.join(saida_teste,"registros_teste.csv"))

# Função para copiar imagens em pastas de classe
def copiar_imagens(df, destino):
    for _, row in df.iterrows():
        identificador = str(row["Registro: Identificador"])
        classe = str(row["Animal: Tipo"]).strip()  # nome da pasta da classe

        # Criar a pasta da classe dentro do destino, se não existir
        destino_classe = os.path.join(destino, classe)
        os.makedirs(destino_classe, exist_ok=True)

        # procurar todas as imagens que começam com esse identificador
        imagens_encontradas = [
            img for img in os.listdir(pasta_imagens)
            if img.startswith(identificador + "_")
        ]

        if imagens_encontradas:
            for img_nome in imagens_encontradas:
                caminho_origem = os.path.join(pasta_imagens, img_nome)
                caminho_destino = os.path.join(destino_classe, img_nome)
                shutil.copy(caminho_origem, caminho_destino)
        else:
            print(f"Aviso: Nenhuma imagem encontrada para ID {identificador}")

# Rodar para cada CSV
copiar_imagens(treino_df, saida_treino)
copiar_imagens(val_df, saida_val)
copiar_imagens(teste_df, saida_teste)

print("Processo concluído!")



'''import os
import shutil
import pandas as pd

# Caminhos
pasta_imagens = "D:/Users/anton/Desktop/sissgeo/imagens"

saida_treino = 'D:/Users/anton/Desktop/imagens_particionadas/Batch1/train/'
saida_val = 'D:/Users/anton/Desktop/imagens_particionadas/Batch1/validation/'
saida_teste = 'D:/Users/anton/Desktop/imagens_particionadas/Batch1/test/'

# Criar pastas de saída se não existirem
os.makedirs(saida_treino, exist_ok=True)
os.makedirs(saida_val, exist_ok=True)
os.makedirs(saida_teste, exist_ok=True)

# CSVs 
treino_df = pd.read_csv(os.path.join(saida_treino,"registros_treino.csv"))
val_df = pd.read_csv(os.path.join(saida_val,"registros_validacao.csv"))
teste_df = pd.read_csv(os.path.join(saida_teste,"registros_teste.csv"))

# Função para copiar imagens de acordo com os registros
def copiar_imagens(df, destino):
    for _, row in df.iterrows():
        identificador = str(row["Registro: Identificador"])

        # procurar todas as imagens que começam com esse identificador
        imagens_encontradas = [
            img for img in os.listdir(pasta_imagens)
            if img.startswith(identificador + "_")
        ]

        if imagens_encontradas:
            for img_nome in imagens_encontradas:
                caminho_origem = os.path.join(pasta_imagens, img_nome)
                shutil.copy(caminho_origem, os.path.join(destino, img_nome))
        else:
            print(f"Aviso: Nenhuma imagem encontrada para ID {identificador}")

# Rodar para cada CSV
copiar_imagens(treino_df, saida_treino)
copiar_imagens(val_df, saida_val)
copiar_imagens(teste_df, saida_teste)

print("Processo concluído!")


'''