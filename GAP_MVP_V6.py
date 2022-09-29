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

def gain(label):

  return tf.math.abs(label)

def _create_feature_dict():
  """Helper function for creating an empty feature dict for defaultdict."""
  return {"embeddings": [], "ranking": []}


def _sample_list(
    feature_lists,
    num_examples_per_list,
    random_state,
):
  """Function for sampling a list example from given feature lists."""
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
      feature_lists["ranking"][idx]
      for idx in sampled_indices
  ]

  return (
      sampled_embeddings,
      tf.concat(sampled_rankings, 0),
  )

def sample_listwise( ranking_dataset, num_list_per_cc, num_examples_per_list, seed=42):
  
  random_state = np.random.RandomState(seed)

  example_lists_by_cc = collections.defaultdict(_create_feature_dict)

  
  for example in ranking_dataset:
    user_id = example["cc_id"].numpy()
    example_lists_by_cc[user_id]["embeddings"].append(
        example["embeddings"])
    example_lists_by_cc[user_id]["ranking"].append(
        example["ranking"])

  tensor_slices = {"cc_id": [], "embeddings": [], "ranking": []}

  for cc_id, feature_lists in example_lists_by_cc.items():
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
      tensor_slices["ranking"].append(sampled_rankings)

  return tf.data.Dataset.from_tensor_slices(tensor_slices)

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
	
	
	tensor_slices = {"cc_id": [], "embeddings": [], "ranking": []}
	for index,row in dataset.iterrows():
		tensor_slices["cc_id"].append(np.array(row["global_cc"]))
		tensor_slices["embeddings"].append(np.array(row[features]))
		tensor_slices["ranking"].append(np.array(row["store_rank"]))
	return tf.data.Dataset.from_tensor_slices(tensor_slices)


def create_observations(element):
	embeddings=element["embeddings"]
	labels=element["ranking"]
	return (embeddings, labels)

batch_size = 2
train_size =10
test_size =10
num_lists_per_cc = 2
num_examples_per_list = 2
raw_dataset_size = 10000

data=create_tensor_dataset(path = '/Users/colinfritz/Desktop/gap_ranking_dataset.csv',read_rows=raw_dataset_size)
data=sample_listwise(ranking_dataset= data,num_list_per_cc= num_lists_per_cc,num_examples_per_list= num_examples_per_list ,seed=42)
data=data.map(create_observations)
shuffled = data.shuffle(1000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(train_size)
test = shuffled.skip(train_size).take(test_size)
cached_train = train.shuffle(1000).batch(batch_size).cache()
cached_test = test.batch(batch_size).cache()

inputs = keras.Input(shape=(2,137))
layer_one=tf.keras.layers.Dense(256, activation="relu")(inputs)
layer_two=tf.keras.layers.Dense(64, activation="relu")(layer_one)
layer_three=tf.keras.layers.Dense(1)(layer_two)
outputs = tf.keras.layers.Reshape((2,))(layer_three)
listwise_model = keras.Model(inputs=inputs, outputs=outputs, name="ranking_model")


epochs = 30
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), 
	loss=tfr.keras.losses.ListMLELoss(), 
	metrics=[tfr.keras.metrics.NDCGMetric(name="ndcg_metric", gain_fn=gain), tf.keras.metrics.RootMeanSquaredError()])
listwise_model.fit(cached_train, epochs=epochs, verbose=False)
listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))
predictions=listwise_model.predict(cached_test)
print("PREDICTIONS: " + str(predictions))
print("RANKINGS: ")
for item in cached_test.as_numpy_iterator():
	print(str(item[1]))

