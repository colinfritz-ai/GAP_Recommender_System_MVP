import pprint
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import collections




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


class RankingModel(tfrs.Model):

  def __init__(self, loss):
    super().__init__()

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

    return self.score_model(features['embeddings'])

  def compute_loss(self, features, training=False):
    labels = features.pop("ranking")

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )





dataset = {"cc_id":[], "embeddings":[], "ranking":[]}
for _ in range(1000):
  embedding=np.random.choice(100,10)
  rank = np.random.choice(100,1).astype('float64')
  cc_id = np.random.choice(10,1)
  # print("embedding: " + str(embedding))
  # print("rank: " + str(rank.shape))
  # print("cc_id: " + str(cc_id))
  dataset["cc_id"].append(str(cc_id[0]))
  dataset["embeddings"].append(embedding)
  dataset["ranking"].append(rank)

  dataset = tf.data.Dataset.from_tensor_slices(dataset)



  listwise_dataset=sample_listwise(ranking_dataset= dataset,
      num_list_per_cc= 1,
      num_examples_per_list= 2,
      seed=42)


  # tf.random.set_seed(42)

  # # Split between train and tests sets
  # shuffled = listwise_dataset.shuffle(10000, seed=42, reshuffle_each_iteration=False)

  # train = shuffled.take(200)
  # test = shuffled.skip(200).take(100)
  print(listwise_dataset.take(1))
# epochs = 30

# cached_train = train.shuffle(1000).batch(10).cache()
# cached_test = test.batch(10).cache()

# mse_model = model.RankingModel(tf.keras.losses.MeanSquaredError())
# mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# mse_model.fit(cached_train, epochs=epochs, verbose=False)

# mse_model_result = mse_model.evaluate(cached_test, return_dict=True)
# print("NDCG of the MSE Model: {:.4f}".format(mse_model_result["ndcg_metric"]))

# listwise_model = model.RankingModel(tfr.keras.losses.ListMLELoss())
# listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# listwise_model.fit(cached_train, epochs=epochs, verbose=False)

# listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
# print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))