import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/features_municipio_anio.csv')
df = df[df['total_nacimientos'] >= 10].copy()

df_arauca = df[df['COD_DPTO'] == 81]

df_2024 = df_arauca[df_arauca['ANO'] == 2024]
weighted_mean = (df_2024['defunciones_fetales'].sum() / df_2024['total_nacimientos'].sum()) * 1000
print(f'2024 Arauca Weighted Mean: {weighted_mean}')

df_2020 = df_arauca[df_arauca['ANO'] == 2020]
weighted_mean_2020 = (df_2020['defunciones_fetales'].sum() / df_2020['total_nacimientos'].sum()) * 1000
print(f'2020 Arauca Weighted Mean: {weighted_mean_2020}')
