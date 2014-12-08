import pandas as pd
import numpy as np

DATA_DIR = '../../../data/'

baseball = pd.read_csv(DATA_DIR + 'baseball.csv', index_col='id')

# Indexing and Selection

player_id = baseball.player + baseball.year.astype(str) + baseball.team
baseball.index = player_id

# Sorting and Ranking

slg = lambda x: (x['h']-x['X2b']-x['X3b']-x['hr'] + 2*x['X2b'] + 3*x['X3b'] + 4*x['hr'])/(x['ab']+1e-6)

baseball['SLG'] = baseball.apply(slg, axis=1)

baseball['SLG_Diff'] = baseball['SLG'] - baseball['SLG'].max()

baseball['SLG_Diff']
