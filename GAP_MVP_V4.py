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

def _sample_list(feature_lists, num_examples_per_list, random_state):
	if random_state is None:
		random_state = np.random.RandomState()

	sampled_indices = random_state.choice(
	  range(len(feature_lists["embeddings"])),
	  size=num_examples_per_list,
	  replace=False,
	)
	sampled_embeddings = [
	  feature_lists["embeddings"][idx] for idx in sampled_indices
	]
	sampled_rankings = [
	  feature_lists["rankings"][idx]
	  for idx in sampled_indices
	]

	return (
	  sampled_embeddings,
	  tf.concat(sampled_rankings,0)
	)
	

def _create_feature_dict():
	return {"embeddings": [], "rankings": []}

# def sample_listwise(features, ranking_dataset, num_list_per_cc, num_examples_per_list, seed):
# 	random_state = np.random.RandomState(seed)

# 	example_lists_by_user = collections.defaultdict(_create_feature_dict)

# 	for index,example in ranking_dataset.iterrows():
# 		cc_id = example["global_cc"]
# 		example_lists_by_user[cc_id]["embeddings"].append(
# 		example[features].to_numpy())
# 		example_lists_by_user[cc_id]["rankings"].append(
# 		example["store_rank"])

# 	tensor_slices = {"cc_id": [], "embeddings": [], "rankings": []}

# 	for cc_id, feature_lists in example_lists_by_user.items():
# 		for _ in range(num_list_per_cc):

# 		  # Drop the user if they don't have enough ratings.
# 		  if len(feature_lists["embeddings"]) < num_examples_per_list:
# 		    continue

# 		  sampled_embeddings, sampled_rankings = _sample_list(
# 		      feature_lists,
# 		      num_examples_per_list,
# 		      random_state=random_state,
# 		  )
# 		  tensor_slices["cc_id"].append(cc_id)
# 		  tensor_slices["embeddings"].append(sampled_embeddings)
# 		  tensor_slices["rankings"].append(sampled_rankings)

# 	return tf.data.Dataset.from_tensor_slices(tensor_slices)
def sample_listwise(features, ranking_dataset, num_list_per_cc, num_examples_per_list, seed):
	random_state = np.random.RandomState(seed)

	example_lists_by_user = collections.defaultdict(_create_feature_dict)

	for example in ranking_dataset:
		cc_id = example["global_cc"]
		example_lists_by_user[cc_id]["embeddings"].append(
		example["embeddings"].to_numpy())
		example_lists_by_user[cc_id]["rankings"].append(
		example["store_rank"])

	tensor_slices = {"cc_id": [], "embeddings": [], "rankings": []}

	for cc_id, feature_lists in example_lists_by_user.items():
		for _ in range(num_list_per_cc):

		  # Drop the user if they don't have enough ratings.
		  if len(feature_lists["embeddings"]) < num_examples_per_list:
		    continue

		  sampled_embeddings, sampled_rankings = _sample_list(
		      feature_lists,
		      num_examples_per_list,
		      random_state=random_state,
		  )
		  tensor_slices["cc_id"].append(cc_id)
		  tensor_slices["embeddings"].append(sampled_embeddings)
		  tensor_slices["rankings"].append(sampled_rankings)

	return tf.data.Dataset.from_tensor_slices(tensor_slices)

dataset = {"cc_id":[], "embeddings":[], "ranking":[]}
for _ in range(1000):
	embedding=np.random.choice(100,10)
	rank = np.random.choice(100,1).astype('float64')
	cc_id = np.random.choice(1000,1)
	dataset["cc_id"].append(str(cc_id[0]))
	dataset["embeddings"].append(embedding)
	dataset["ranking"].append(rank)
print("manufactued dataset numpy version: " + str(dataset))
dataset = tf.data.Dataset.from_tensor_slices(dataset)

listwise_dataset=gp.sample_listwise(ranking_dataset= dataset,
    num_list_per_cc= 2,
    num_examples_per_list= 2,
    seed=42)
print("Manufactured dataset: " + str(listwise_dataset))

target_variable = "loc_cc_target_demand"
dataset = pd.read_csv ('/Users/colinfritz/Desktop/Sample_Tensorflow_Ranking_Dataset.csv')
print("number of unique ccs: " + str(len(dataset["global_cc"].unique())))
dataset = dataset.select_dtypes(np.number)
dataset['store_rank'] = dataset.groupby('global_cc')[target_variable].rank('first')
# dataset = dataset.fillna(0)
features = [x for x in dataset.columns.tolist() if x not in [target_variable, "global_cc", "sell_loc_str_nbr", "store_rank", "Baby_Shop_Count", "Newborn_Shop_Count"]]
for column in features:
	dataset[column] = (dataset[column]-dataset[column].mean())/dataset[column].std()
dataset=dataset.drop(["Baby_Shop_Count", "Newborn_Shop_Count", "sell_loc_str_nbr", target_variable],axis=1)
dataset=dataset.fillna(0)
tensor_slices = {"cc_id": [], "embeddings": [], "ranking": []}
for index,row in dataset.iterrows():
	tensor_slices["cc_id"].append(np.array(row["global_cc"]))
	tensor_slices["embeddings"].append(np.array(row[features]))
	tensor_slices["ranking"].append(np.array(row["store_rank"]))
print("TENSOR SLICES RANKING: " + str(tensor_slices["ranking"][:10]))
print("TENSOR SLICES EMBEDDINGS: " + str(tensor_slices["embeddings"][:10]))
print("TENSOR SLICES CC_ID: " + str(tensor_slices["cc_id"][:10]))

dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
listwise_dataset=gp.sample_listwise(ranking_dataset= dataset,
    num_list_per_cc= 2,
    num_examples_per_list= 2,
    seed=42)
# listwise_dataset=gp.sample_listwise(features, dataset, 2, 2, 42)

print("Read in from desktop dataset: " + str(listwise_dataset))




class CustomModel(tfrs.Model):
	def __init__(self,loss):
		super().__init__()
		self.layer_one =  tf.keras.layers.Dense(256, activation="relu")
		self.layer_two = tf.keras.layers.Dense(64, activation="relu")
		self.layer_three = tf.keras.layers.Dense(1)

		self.task = tfrs.tasks.Ranking(loss=loss, metrics=[
		tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
		tf.keras.metrics.RootMeanSquaredError()])

	def call(self, inputs):
		x = self.layer_one(inputs['embeddings'])
		x = self.layer_two(x)
		x = self.layer_three(x)
		return tf.squeeze(x,axis=-1)

	def compute_loss(self, features, training=True):
	    labels = features.pop("ranking")
	    scores = self(features)
	    return self.task(
	        labels=labels,
	        predictions=scores
    )



tf.random.set_seed(42)

# Split between train and tests sets, as before.
shuffled = listwise_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(100)
test = shuffled.skip(100).take(10)

epochs = 30

cached_train = train.shuffle(1000).batch(2).cache()
cached_test = test.batch(2).cache()

print("cached dataset size: " + str(len(cached_train)))

print("cached dataset element check: " + str(cached_train.take(1)))





mse_model = CustomModel(tf.keras.losses.MeanSquaredError())
mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

mse_model.fit(cached_train, epochs=epochs, verbose=False)

mse_model_result = mse_model.evaluate(cached_test, return_dict=True)
print("NDCG of the MSE Model: {:.4f}".format(mse_model_result["ndcg_metric"]))


listwise_model = CustomModel(tfr.keras.losses.ListMLELoss())
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.001))
listwise_model.fit(cached_train, epochs=epochs, verbose=False)
listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))