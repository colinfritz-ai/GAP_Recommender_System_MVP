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



class RankingModel(tfrs.Model):

  def __init__(self, loss):
    super().__init__()

    self.embeddings = tf.keras.Sequential()
    # Compute predictions.
    self.score_model = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=loss,
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def call(self, features):
    feature_embeddings = self.embeddings(features["embeddings"])
    # pprint.pprint("feature shape check: " + str(features))
    return self.score_model(feature_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("rankings")
    cc_id = features.pop("cc_id")
    # pprint.pprint("labels shape check: " + str(labels))
    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )



target_variable = "loc_cc_target_demand"
dataset = pd.read_csv ('/Users/colinfritz/Desktop/Sample_Tensorflow_Ranking_Dataset.csv')
dataset = dataset.select_dtypes(np.number)
dataset['store_rank'] = dataset.groupby('global_cc')[target_variable].rank('first')
dataset = dataset.fillna(0)
features = [x for x in dataset.columns.tolist() if x not in [target_variable, "global_cc", "sell_loc_str_nbr", "store_rank"]]
for column in features:
	dataset[column] = (dataset[column]-dataset[column].mean())/dataset[column].std()

dataset=sample_listwise(features, dataset, 1, 1, 42)
# for example in dataset.take(1):
# 	pprint.pprint(example)

tf.random.set_seed(42)

# Split between train and tests sets, as before.
shuffled = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(200)
test = shuffled.skip(200).take(100)

epochs = 30

cached_train = train.shuffle(10000).batch(16).cache()
cached_test = test.batch(16).cache()

listwise_model = RankingModel(tfr.keras.losses.ListMLELoss())
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.000001))
listwise_model.fit(cached_train, epochs=epochs, verbose=False)
listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))
