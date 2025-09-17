import os
import pandas as pd

saida = "D:/Users/anton/Desktop/imagens_separadas_completas"

# List folders
dados = []
total = 0
for pasta in os.listdir(saida):
    caminho = os.path.join(saida, pasta)
    if os.path.isdir(caminho):
        qtd = len([f for f in os.listdir(caminho) if os.path.isfile(os.path.join(caminho, f))])
        dados.append({"Classe": pasta, "Quantidade": qtd})
        total += qtd

# Create DataFrame
df_classes = pd.DataFrame(dados).sort_values("Quantidade", ascending=False).reset_index(drop=True)

# Add total
df_classes = pd.concat([df_classes, pd.DataFrame([{"Classe": "TOTAL", "Quantidade": total}])], ignore_index=True)

print(df_classes)

# Save as CSV
df_classes.to_csv("C:/Users/anton/projects/SISSGeo/resumo_animais_imagens.csv", index=False, encoding="utf-8-sig")