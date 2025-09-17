import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ler o CSV
df = pd.read_csv("D:/Users/anton/Desktop/sissgeo/registros.csv")

coluna_classe = "Animal: Tipo"
coluna_grupo = "Registro: Identificador"

# Contar quantos registros por classe
contagens = df[coluna_classe].value_counts()

# Manter apenas classes com >= 100 registros
df_filtrado = df[df[coluna_classe].isin(contagens[contagens >= 100].index)].copy()
df_rejeitado = df[df[coluna_classe].isin(contagens[contagens < 100].index)].copy()


print("Tamanho original:", len(df))
print("Tamanho filtrado:", len(df_filtrado))
print("Tamanho rejeitado:", len(df_rejeitado))
print("Classes restantes:", df_filtrado[coluna_classe].nunique())


df_rejeitado[coluna_classe] = "Outro"

df_filtrado = pd.concat([df_filtrado, df_rejeitado], ignore_index=True).copy()

print("Tamanho final:", len(df_filtrado))
print("Classes presentes:", df_filtrado[coluna_classe].nunique())



# ====== Remover duplicatas de Identificador ======
df_unicos = df_filtrado.drop_duplicates(subset=coluna_grupo, keep="first")
df_dup = df_filtrado[df_filtrado.duplicated(subset=coluna_grupo, keep="first")]

print("Registros únicos:", len(df_unicos))
print("Registros duplicados:", len(df_dup))

# ====== Separação normal com estratificação ======
train_df, temp_df = train_test_split(df_unicos, test_size=0.30, stratify=df_unicos[coluna_classe], random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[coluna_classe], random_state=42)

# ====== Recolocar duplicados nos conjuntos ======

# Separar duplicados conforme seus ids
dup_train = df_dup[df_dup[coluna_grupo].isin(train_df[coluna_grupo])]
dup_val   = df_dup[df_dup[coluna_grupo].isin(val_df[coluna_grupo])]
dup_test  = df_dup[df_dup[coluna_grupo].isin(test_df[coluna_grupo])]

# Concatenar
train_df = pd.concat([train_df, dup_train])
val_df   = pd.concat([val_df, dup_val])
test_df  = pd.concat([test_df, dup_test])

# Conferindo proporções finais
print("Treino:", len(train_df))
print("Validação:", len(val_df))
print("Teste:", len(test_df))

# ====== Validação extra: nenhum id em mais de um conjunto ======
ids_train = set(train_df[coluna_grupo])
ids_val   = set(val_df[coluna_grupo])
ids_test  = set(test_df[coluna_grupo])

intersecao = (ids_train & ids_val) | (ids_train & ids_test) | (ids_val & ids_test)
print("Identificadores repetidos entre conjuntos:", intersecao)

# ===== salvar =====
output_train_dir = 'D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/train/'
output_val_dir = 'D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/validation/'
output_test_dir = 'D:/Users/anton/Desktop/imagens_particionadas_outros/Batch1/test/'
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_train_dir, "registros_treino.csv"), index=False)
val_df.to_csv(os.path.join(output_val_dir, "registros_validacao.csv"), index=False)
test_df.to_csv(os.path.join(output_test_dir, "registros_teste.csv"), index=False)
