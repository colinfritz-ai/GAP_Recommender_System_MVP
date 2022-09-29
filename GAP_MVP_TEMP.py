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
	  tf.concat(sampled_rankings, 0),
	)
	

def _create_feature_dict():
	return {"embeddings": [], "rankings": []}

def sample_listwise(features, ranking_dataset, num_list_per_cc, num_examples_per_list, seed):
	random_state = np.random.RandomState(seed)

	example_lists_by_user = collections.defaultdict(_create_feature_dict)

	for index,example in ranking_dataset.iterrows():
		cc_id = example["global_cc"]
		example_lists_by_user[cc_id]["embeddings"].append(
		example[features].to_numpy())
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


target_variable = "loc_cc_target_demand"
dataset = pd.read_csv ('/Users/colinfritz/Desktop/Sample_Tensorflow_Ranking_Dataset.csv')
dataset = dataset.select_dtypes(np.number)
dataset['ranking'] = dataset.groupby('global_cc')[target_variable].rank('first')
dataset = dataset.fillna(0)
features = [x for x in dataset.columns.tolist() if x not in [target_variable, "global_cc", "sell_loc_str_nbr", "store_rank"]]
for column in features:
	dataset[column] = (dataset[column]-dataset[column].mean())/dataset[column].std()

# dataset=sample_listwise(features, dataset, 2, 2, 42)
# listwise_dataset = dataset 

# dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset=dataset.drop(["Baby_Shop_Count", "Newborn_Shop_Count", "sell_loc_str_nbr", target_variable],axis=1)
dataset = dataset.rename(columns ={"global_cc": "cc_id"})
listwise_dataset=gp.sample_listwise(ranking_dataset= dataset,
    num_list_per_cc= 1,
    num_examples_per_list= 1,
    seed=42)


tf.random.set_seed(42)

# Split between train and tests sets
shuffled = listwise_dataset.shuffle(10000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(200)
test = shuffled.skip(200).take(100)

epochs = 30

cached_train = train.shuffle(1000).batch(10).cache()
cached_test = test.batch(10).cache()

mse_model = model.RankingModel(tf.keras.losses.MeanSquaredError())
mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

mse_model.fit(cached_train, epochs=epochs, verbose=False)

mse_model_result = mse_model.evaluate(cached_test, return_dict=True)
print("NDCG of the MSE Model: {:.4f}".format(mse_model_result["ndcg_metric"]))

listwise_model = model.RankingModel(tfr.keras.losses.ListMLELoss())
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

listwise_model.fit(cached_train, epochs=epochs, verbose=False)

listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))