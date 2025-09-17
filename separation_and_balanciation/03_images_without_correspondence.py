import os
import pandas as pd
from python_calamine import CalamineWorkbook


# Paths
pasta_imagens = "D:/Users/anton/Desktop/sissgeo/imagens"

df = pd.read_csv("D:/Users/anton/Desktop/sissgeo/registros.csv")


# Normalize records
df["Registro: Identificador"] = df["Registro: Identificador"].astype(str)
registros_validos = set(df["Registro: Identificador"])


imagens_sem_correspondencia = []

# List images without correspondence
for img in os.listdir(pasta_imagens):
    if "_" not in img:
        continue
    registro = img.split("_")[0].strip()
    if registro not in registros_validos:
        imagens_sem_correspondencia.append(img)


df_sem_correspondencia = pd.DataFrame(imagens_sem_correspondencia, columns=["Imagem"])

print(df_sem_correspondencia)
print(f"\nTotal de imagens sem correspondência: {len(imagens_sem_correspondencia)}")


df_sem_correspondencia.to_csv("C:/Users/anton/projects/SISSGeo/imagens_sem_correspondencia.csv", index=False, encoding="utf-8-sig")