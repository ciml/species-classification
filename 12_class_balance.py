import os
import random
import shutil

def balancear_classes(pasta_treino, limite=2000, pasta_saida="treino_balanceado", seed=42):
    os.makedirs(pasta_saida, exist_ok=True)
    random.seed(seed)
    # Itera pelas classes (pastas dentro da pasta de treino)
    for classe in os.listdir(pasta_treino):
        caminho_classe = os.path.join(pasta_treino, classe)
        if not os.path.isdir(caminho_classe):
            continue
        
        imagens = [img for img in os.listdir(caminho_classe) 
                   if os.path.isfile(os.path.join(caminho_classe, img))]
        
        qtd = len(imagens)
        print(f"Classe '{classe}': {qtd} imagens")

        # Cria pasta de saída para a classe
        pasta_destino = os.path.join(pasta_saida, classe)
        os.makedirs(pasta_destino, exist_ok=True)

        if qtd > limite:
            # Sorteia 2000 imagens (mantém só elas)
            escolhidas = random.sample(imagens, limite)
            extras = []
        elif qtd < limite:
            # Copia todas originais
            escolhidas = imagens[:]
            # Gera duplicatas aleatórias
            extras = random.choices(imagens, k=limite - qtd)
        else:
            escolhidas = imagens
            extras = []

        # Copia imagens originais
        for img in escolhidas:
            origem = os.path.join(caminho_classe, img)
            destino = os.path.join(pasta_destino, img)
            shutil.copy2(origem, destino)

        # Copia duplicatas com sufixo
        for i, img in enumerate(extras, start=1):
            origem = os.path.join(caminho_classe, img)
            nome, ext = os.path.splitext(img)
            nome_novo = f"aug_{i:05d}_" + os.path.basename(img)
            destino = os.path.join(pasta_destino, nome_novo)
            shutil.copy2(origem, destino)

        print(f"--> Classe '{classe}' balanceada para {limite} imagens.\n")


            



balancear_classes("D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/train/", limite=2000, pasta_saida="D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/train_balanced")
