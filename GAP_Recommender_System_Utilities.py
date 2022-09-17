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