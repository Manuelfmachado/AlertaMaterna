import pandas as pd

df = pd.read_csv('../data/processed/features_alerta_materna.csv')

print('TODAS LAS FEATURES EXISTENTES:')
print(f'Total: {len(df.columns)} features\n')

for i, col in enumerate(sorted(df.columns), 1):
    print(f'{i:2d}. {col}')
