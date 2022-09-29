import pprint
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import collections
import GAP_Recommender_System_Model as model
import GAP_Recommender_System_Utilities as gp
import pandas as pd


target_variable = "loc_cc_target_demand"
dataset = pd.read_csv ('/Users/colinfritz/Desktop/Sample_Tensorflow_Ranking_Dataset.csv')
print(len(dataset["global_cc"].unique()))
dataset = dataset.select_dtypes(np.number)
dataset['store_rank'] = dataset.groupby('global_cc')[target_variable].rank('first')
# dataset = dataset.fillna(0)
features = [x for x in dataset.columns.tolist() if x not in [target_variable, "global_cc", "sell_loc_str_nbr", "store_rank"]]
for column in features:
	dataset[column] = (dataset[column]-dataset[column].mean())/dataset[column].std()

for column in dataset.columns.tolist():
	dataset[column].fillna(value=dataset[column].mean(), inplace=True)
	print(column + " has this many nulls: " + str(dataset[column].isnull().sum()))



# dataset=sample_listwise(features, dataset, 1, 2, 42)