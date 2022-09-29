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

def sample_listwise(
    ranking_dataset,
    num_list_per_cc,
    num_examples_per_list,
    seed,
):
  """Function for converting the rankings dataset to a listwise dataset.
  Args:
      ranking_dataset:
        The training dataset with [CC,embeddinga,rank] for the specified time period
      num_list_per_cc:
        An integer representing the number of lists that should be sampled for
        each cc in the training dataset.
      num_examples_per_list:
        An integer representing the number of store ranks to be sampled for each list
        from the list of stores ranked "by" the cc.  Like a user ranking movies.    
      seed:
        An integer for creating `np.random.RandomState.
  Returns:
      A tf.data.Dataset containing list examples.
      Each example contains three keys: "cc_id", "embeddings", and
      "ranking". "cc_id" maps to a integer tensor that represents the
      cc_id for the example. "embeddings" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.Tensor. It represents the list
      of store,cc embedding descriptions. "ranking" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.float32. It represents the
      ranking of each store attached to the cc_id in the candidate list.
  """
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
	print("UNIQUE CCS: " + str(len(dataset["global_cc"].unique())))
	for item in dataset.columns.tolist():
		print("NULLS PRESENT IN " + item + ": " + str(dataset[item][dataset[item]==0].count()))
		print("INF PRESENT IN " + item + ": " + str(np.isinf(dataset[item]).values.sum()))
		print("DATA TYPE: " + str(dataset.dtypes[item]))
	tensor_slices = {"cc_id": [], "embeddings": [], "ranking": []}
	for index,row in dataset.iterrows():
		tensor_slices["cc_id"].append(np.array(row["global_cc"]))
		tensor_slices["embeddings"].append(np.array(row[features]))
		tensor_slices["ranking"].append(np.array(row["store_rank"]))
	return tf.data.Dataset.from_tensor_slices(tensor_slices)

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




dataset=create_tensor_dataset( path = '/Users/colinfritz/Desktop/gap_ranking_dataset.csv',read_rows=10000)
# path = '/Users/colinfritz/Desktop/gap_ranking_dataset.csv'
logdir="/Users/colinfritz/Desktop/my_repos/GAP_Recommender_System_MVP/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
listwise_dataset=sample_listwise(ranking_dataset= dataset,
    num_list_per_cc= 2,
    num_examples_per_list= 2,
    seed=42)

print("LISTWISE_SIZE: " + str(len(listwise_dataset)))
tf.random.set_seed(42)

# Split between train and tests sets, as before.
shuffled = listwise_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=False)
print("SHUFFLED_SIZE: " + str(len(shuffled)))
train = shuffled.take(10)
test = shuffled.skip(10).take(10)

epochs = 30
print("TRAIN_SIZE: " + str(len(train)))
print("TEST_SIZE: " + str(len(test)))
cached_train = train.shuffle(1000).batch(2).cache()
cached_test = test.batch(2).cache()
print("CACHED_TEST_SIZE: " + str(len(cached_test)))

mse_model = CustomModel(tf.keras.losses.MeanSquaredError())
mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

mse_model.fit(cached_train, epochs=epochs, verbose=False)

mse_model_result = mse_model.evaluate(cached_test, return_dict=True)
print("NDCG of the MSE Model: {:.4f}".format(mse_model_result["ndcg_metric"]))


listwise_model = CustomModel(tfr.keras.losses.ListMLELoss())
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
listwise_model.fit(cached_train, epochs=epochs, verbose=False)
listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))

#prediction test 
print("TRAIN_PREDICTIONS: " + str(listwise_model.predict(cached_train)))
print("TEST_PREDICTIONS: " + str(listwise_model.predict(cached_test)))
print("TRAIN NANS PREDICTED?: " + str(np.isinf(listwise_model.predict(cached_train)).any()))
print("TEST NANS PREDICTED?: " + str(np.isinf(listwise_model.predict(cached_test)).any()))
count=0
for obs in cached_test.as_numpy_iterator():
	print("OBS: " + str(obs))
	for item in obs["embeddings"]:
		print("TEST_CONTENTS_NAN: " + str(max(item[0])))
		print("TEST_CONTENTS_NAN: " + str(max(item[1])))

for item in listwise_model.predict(cached_test):
	print("TEST PREDICTED MAX: " + str(max(item)))
