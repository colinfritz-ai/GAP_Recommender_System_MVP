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
from datetime import datetime
from tensorflow import keras

# ratings = tfds.load("movielens/100k-ratings", split="train")
# movies = tfds.load("movielens/100k-movies", split="train")

# # ratings = ratings.map(lambda x: {
# #     "movie_title": x["movie_title"],
# #     "user_id": x["user_id"],
# #     "user_rating": x["user_rating"],
# # })
# print("RATINGS DATA: " + str(ratings))
def create_tensor_dataset(target_variable="loc_cc_target_demand", path='/Users/colinfritz/Desktop/Sample_Tensorflow_Ranking_Dataset.csv', read_rows=10000):
	dataset = pd.read_csv(path, nrows=read_rows)
	dataset = dataset.select_dtypes(np.number)
	dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
	dataset['store_rank'] = dataset.groupby('global_cc')[target_variable].rank('first')

	features = [x for x in dataset.columns.tolist() if x not in [target_variable, "global_cc", "sell_loc_str_nbr", "store_rank", "Baby_Shop_Count", "Newborn_Shop_Count"]]
	for column in features:
		dataset[column] = (dataset[column]-dataset[column].mean())/dataset[column].std()
	dataset=dataset.drop(["Baby_Shop_Count", "Newborn_Shop_Count", "sell_loc_str_nbr", target_variable], axis=1)
	dataset=dataset.dropna()
	print("UNIQUE CCS: " + str(len(dataset["global_cc"].unique())))
	# for item in dataset.columns.tolist():
	# 	# print("NULLS PRESENT IN " + item + ": " + str(dataset[item][dataset[item]==0].count()))
	# 	print("INF PRESENT IN " + item + ": " + str(np.isinf(dataset[item].values).sum()))
		# print("DATA TYPE: " + str(dataset.dtypes[item]))
	tensor_slices = {"cc_id": [], "embeddings": [], "ranking": []}
	for index,row in dataset.iterrows():
		tensor_slices["cc_id"].append(np.array(row["global_cc"]))
		tensor_slices["embeddings"].append(np.array(row[features]))
		tensor_slices["ranking"].append(np.array(row["store_rank"]))
	# print("TENSOR_SLICES EMBEDDINGS: " + str(tensor_slices["embeddings"][:10]))
	# print("TENSOR_SLICES CC_ID: " + str(tensor_slices["cc_id"][:10]))
	# print("TENSOR_SLICES RANKING: " + str(tensor_slices["ranking"][:10]))
	return tf.data.Dataset.from_tensor_slices(tensor_slices)

data_one=create_tensor_dataset(read_rows=10000)
print("TENSORFLOW DATASET ONE: " + str(data_one))
print("SEPARATION")
print("SEPARATION")
print("SEPARATION")
print("SEPARATION")
data_two=create_tensor_dataset(path = '/Users/colinfritz/Desktop/gap_ranking_dataset.csv', read_rows=10000)
print("TENSORFLOW DATASET TWO: " + str(data_two))


y_true = [[213, 211]]
y_pred = [[9, 1]]
def gain(label):
  """Computes `2**x - 1` element-wise for each label.
  Can be used to define `gain_fn` for `tfr.keras.metrics.NDCGMetric`.
  Args:
    label: A `Tensor` or anything that can be converted to a tensor using
      `tf.convert_to_tensor`.
  Returns:
    A `Tensor` that has each input element transformed as `x` to `2**x - 1`.
  """
  return tf.math.abs(label)
ndcg = tfr.keras.metrics.NDCGMetric(gain_fn=gain)
print("RANKING SCORE TEST: " + str(ndcg(y_true, y_pred).numpy()))
