#from python_calamine import CalamineWorkbook

'''
# Read ods with metadata
wb = CalamineWorkbook.from_path("D:/Users/anton/Desktop/ssisgeo/registros_siss-geo_20200114.ods")
data = wb.get_sheet_by_name(wb.sheet_names[0]).to_python()
df = pd.DataFrame(data[1:], columns=data[0])  # first line as header
'''

import pandas as pd

df = pd.read_csv("D:/Users/anton/Desktop/sissgeo/registros.csv")

# Check columns
print(df.columns)

# Group by "Animal: Tipo" and count
resumo = df.groupby("Animal: Tipo").size().reset_index(name="Contagem")

# Order by count
resumo = resumo.sort_values("Contagem", ascending=False).reset_index(drop=True)

# Total records
total_registros = resumo["Contagem"].sum()
resumo = pd.concat([resumo, pd.DataFrame([{"Animal: Tipo": "TOTAL", "Contagem": total_registros}])], ignore_index=True)


print(resumo)


# Save as CSV
resumo.to_csv("C:/Users/anton/projects/SISSGeo/resumo_animais_dados.csv", index=False, encoding="utf-8-sig")

