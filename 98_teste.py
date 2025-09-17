import os
import pandas as pd

def listar_arquivos_em_df(diretorio_raiz):
    arquivos = []
    for _, _, nomes in os.walk(diretorio_raiz):
        for nome in nomes:
            arquivos.append(nome)  # apenas o nome do arquivo

    df = pd.DataFrame(arquivos, columns=["arquivo_completo"])
    # Extrair apenas o texto antes do primeiro "_"
    df["arquivo"] = df["arquivo_completo"].str.split("_").str[0]
    return df



# Exemplo de uso
df_arquivos = listar_arquivos_em_df("D:/Users/anton/Desktop/sissgeo/imagens/")

df_arquivos["arquivo"] = df_arquivos["arquivo"].astype(int)

# Ler o CSV
df = pd.read_csv("D:/Users/anton/Desktop/sissgeo/registros.csv")

coluna_classe = "Animal: Tipo"
coluna_grupo = "Registro: Identificador"

# Contar quantos registros por classe
contagens = df[coluna_classe].value_counts()

# Manter apenas classes com >= 100 registros
df_filtrado = df[df[coluna_classe].isin(contagens[contagens >= 100].index)].copy()
df_rejeitado = df[df[coluna_classe].isin(contagens[contagens < 100].index)].copy()




df_rejeitado[coluna_classe] = "Outro"

df_filtrado = pd.concat([df_filtrado, df_rejeitado], ignore_index=True).copy()





df_filtrado[coluna_grupo] = df_filtrado[coluna_grupo].astype(int)

print(df_arquivos.head())
print(df_filtrado.head())

df_join = pd.merge(df_filtrado, df_arquivos, right_on="arquivo", left_on=coluna_grupo, how="inner")
print(df_join.head(100))

df_join = df_join.drop_duplicates(subset=["arquivo_completo", coluna_classe])


print(len(df_join))