'''import os
from collections import defaultdict
from tqdm import tqdm  # pip install tqdm

def encontrar_duplicados_por_nome(diretorio_raiz):
    nome_map = defaultdict(list)

    # Listar todos os arquivos
    arquivos_lista = []
    for raiz, _, arquivos in os.walk(diretorio_raiz):
        for nome in arquivos:
            arquivos_lista.append((raiz, nome))

    # Percorrer com barra de progresso
    for raiz, nome in tqdm(arquivos_lista, desc="Verificando nomes de arquivos"):
        nome_map[nome].append(os.path.join(raiz, nome))

    # Filtrar apenas os nomes duplicados
    duplicados = {nome: caminhos for nome, caminhos in nome_map.items() if len(caminhos) > 1}

    # Imprimir resultados
    total = sum(len(c) - 1 for c in duplicados.values())
    print(f"\nEncontrados {total} arquivos duplicados por nome em {len(duplicados)} grupos:\n")
    for nome, caminhos in duplicados.items():
        print("----")
        print(f"Nome duplicado: {nome}")
        for c in caminhos:
            print(c)

    return duplicados



# Exemplo de uso
duplicados = encontrar_duplicados_por_nome("D:/Users/anton/Desktop/imagens_particionadas/Batch1/")
'''

import os
from collections import defaultdict
from tqdm import tqdm  # pip install tqdm

def encontrar_duplicados_por_nome(diretorio_raiz):
    nome_map = defaultdict(list)

    # Listar todos os arquivos
    arquivos_lista = []
    for raiz, _, arquivos in os.walk(diretorio_raiz):
        for nome in arquivos:
            arquivos_lista.append((raiz, nome))

    # Percorrer com barra de progresso
    for raiz, nome in tqdm(arquivos_lista, desc="Verificando nomes de arquivos"):
        nome_map[nome].append(os.path.join(raiz, nome))

    # Filtrar apenas os nomes duplicados
    duplicados = {nome: caminhos for nome, caminhos in nome_map.items() if len(caminhos) > 1}

    # Imprimir resultados
    total = sum(len(c) - 1 for c in duplicados.values())
    print(f"\nEncontrados {total} arquivos duplicados por nome em {len(duplicados)} grupos:\n")

    # Mostrar grupos de duplicados
    for nome, caminhos in duplicados.items():
        # Diretório pai imediato da imagem
        pastas = [os.path.dirname(c) for c in caminhos]
        # Diretório antes da pasta da imagem (avô)
        pastas_anteriores = [os.path.dirname(p) for p in pastas]

        mesma_pasta_anterior = all(p == pastas_anteriores[0] for p in pastas_anteriores)

        #print(f"Arquivo duplicado: {nome}")
        #for caminho in caminhos:
            #print(f"  -> {caminho}")
        #print(f"  >>> Todos no mesmo diretório anterior à pasta da imagem? {mesma_pasta_anterior}")
        if not mesma_pasta_anterior:
            print(f"      Diretórios diferentes encontrados: {set(pastas_anteriores)}")
        #print("-" * 60)

    return duplicados


# Exemplo de uso
duplicados = encontrar_duplicados_por_nome("D:/Users/anton/Desktop/imagens_particionadas/Batch1/")
