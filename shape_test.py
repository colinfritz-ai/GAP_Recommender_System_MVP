import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import GAP_Recommender_System_Utilities as gp


# dataset = {"cc_id":[], "embeddings":[], "ranking":[]}
# for _ in range(1000):
# 	embedding=np.random.choice(100,10)
# 	rank = np.random.choice(100,1)
# 	cc_id = np.random.choice(1000,1)
# 	dataset["cc_id"].append(str(cc_id))
# 	dataset["embeddings"].append(embedding)
# 	dataset["ranking"].append(rank)

# dataset = tf.data.Dataset.from_tensor_slices(dataset)

# listwise_dataset=gp.sample_listwise(ranking_dataset= dataset,
#     num_list_per_cc= 10,
#     num_examples_per_list= 5,
#     seed=42)
# for example in listwise_dataset.take(1):
# 	print("embedding: " + str(example["embeddings"].numpy()))
# 	print("ranking: " + str(example["ranking"].numpy()))
# 	print("cc_id: " + str(example["cc_id"].numpy()))

# for example in listwise_dataset.take(1):
# 	print("example: " + str(example))

tf.random.set_seed(42)
y_true = [[1., 0.]]
y_pred = [[0.6, 0.8]]
loss = tfr.keras.losses.ListMLELoss()
loss(y_true, y_pred).numpy()

