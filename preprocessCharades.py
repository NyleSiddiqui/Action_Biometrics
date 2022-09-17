import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def splitActions(df):
	df.drop(['quality', 'relevance', 'script', 'objects', 'descriptions', 'verified'], axis=1, inplace=True)
	df = df.dropna(subset = ['actions'])
	new_df_rows = []
	for count, row in enumerate(df.to_numpy()):
		try:
			ast = row[3].split(';')
		except AttributeError:
			ast = row[3]
		for actions in ast:
			action, start, end = actions.split(' ')
			new_df_rows.append([row[0], row[1], row[2], action[1:], start, end])	
	new_df = pd.DataFrame(new_df_rows, columns=['id', 'subject', 'scene', 'action', 'start', 'end'])
	new_df.reset_index()
	new_df.to_csv("CharadesEgo_test.csv")

if __name__ == '__main__':
	pd.options.display.width = 0
	traindf = pd.read_csv("CharadesEgo_train.csv")
	testdf = pd.read_csv('CharadesEgo_test.csv')
	traindf.to_csv('CharadesEgo_train.txt', header=None, index=None, sep=';', mode='a')
	testdf.to_csv('CharadesEgo_test.txt', header=None, index=None, sep=';', mode='a')




