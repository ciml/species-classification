import os
import shutil
import pandas as pd
#from python_calamine import CalamineWorkbook
import re


# Paths
pasta_imagens = "D:/Users/anton/Desktop/sissgeo/imagens"
saida = "D:/Users/anton/Desktop/imagens_separadas_completas"

# Read ods with metadata
df = pd.read_csv("D:/Users/anton/Desktop/sissgeo/registros.csv")




def sanitizar_nome_pasta(nome):
    """Mantém apenas letras, números, espaço, hífen e underscore. Troca o resto por '_'"""
    return re.sub(r'[\\/:*?"<>|]', "_", nome)

# Normalize columns
df["Animal: Tipo"] = df["Animal: Tipo"].fillna("NULL").astype(str).str.strip()
df.loc[df["Animal: Tipo"] == "", "Animal: Tipo"] = "NULL"

# Sanitize classes
df["Animal: Tipo"] = df["Animal: Tipo"].apply(sanitizar_nome_pasta)
df["Registro: Identificador"] = df["Registro: Identificador"].astype(str)

# Creat dictionary
mapa_registro_classe = dict(zip(df["Registro: Identificador"], df["Animal: Tipo"]))


os.makedirs(saida, exist_ok=True)

# Process images
for img in os.listdir(pasta_imagens):
    if "_" not in img:
        continue

    registro = img.split("_")[0].strip()
    classe = mapa_registro_classe.get(registro, "NULL")
  
    destino_classe = os.path.join(saida, classe)
    os.makedirs(destino_classe, exist_ok=True)

    origem = os.path.join(pasta_imagens, img)
    destino = os.path.join(destino_classe, img)

    shutil.copy2(origem, destino) 

print("Imagens copiadas por classe!")