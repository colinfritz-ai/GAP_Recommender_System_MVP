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
# tf.debugging.experimental.enable_dump_debug_info(
#     "/Users/colinfritz/Desktop/my_repos/GAP_Recommender_System_MVP/mvp_logs_v1",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1)
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
	  sampled_rankings
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





class CustomModel(keras.Model):
	def __init__(self):
		super(CustomModel, self).__init__()
		self.layer_one =  tf.keras.layers.Dense(256, activation="relu")
		self.layer_two = tf.keras.layers.Dense(64, activation="relu")
		self.layer_three = tf.keras.layers.Dense(1)

	def train_step(self, data):
	# Unpack the data. Its structure depends on your model and
	# on what you pass to `fit()`.
		y = data["rankings"]

		with tf.GradientTape() as tape:
			y_pred = self(data, training=True) # Forward pass
			# Compute the loss value
			# (the loss function is configured in `compile()`)
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(y, y_pred)
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}

	def call(self, inputs):
		inputs=inputs["embeddings"]
		x=self.layer_one(inputs)
		x=self.layer_two(x)
		x=self.layer_three(x)
		return tf.squeeze(x,axis=-1)


target_variable = "loc_cc_target_demand"
dataset = pd.read_csv ('/Users/colinfritz/Desktop/Sample_Tensorflow_Ranking_Dataset.csv')
print("number of unique ccs: " + str(len(dataset["global_cc"].unique())))
dataset = dataset.select_dtypes(np.number)
dataset['store_rank'] = dataset.groupby('global_cc')[target_variable].rank('first')
# dataset = dataset.fillna(0)
features = [x for x in dataset.columns.tolist() if x not in [target_variable, "global_cc", "sell_loc_str_nbr", "store_rank", "Baby_Shop_Count", "Newborn_Shop_Count"]]
for column in features:
	dataset[column] = (dataset[column]-dataset[column].mean())/dataset[column].std()
dataset=dataset.drop(["Baby_Shop_Count", "Newborn_Shop_Count"],axis=1)

dataset=sample_listwise(features, dataset, 2, 2, 42)
for example in dataset.take(1):
	print(" ")
	print("listwise datset element")
	pprint.pprint(example)
	print("end listwise element check")
	print(" ")
print("dataset size: " + str(len(dataset)))
tf.random.set_seed(42)

# Split between train and tests sets, as before.
shuffled = dataset.shuffle(1000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(100)
test = shuffled.skip(100).take(10)

epochs = 30

cached_train = train.shuffle(1000).batch(2).cache()
cached_test = test.batch(2).cache()

print("cached dataset size: " + str(len(cached_train)))

print("cached dataset element check: " + str(cached_train.take(1)))

logdir="/Users/colinfritz/Desktop/my_repos/GAP_Recommender_System_MVP/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

listwise_model = CustomModel()
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.001), loss=tfr.keras.losses.ListMLELoss(), metrics=[tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()])
listwise_model.fit(cached_train, epochs=epochs, verbose=False, callbacks = [tensorboard_callback])
listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))