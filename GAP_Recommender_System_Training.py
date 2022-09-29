import pprint
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import collections
import GAP_Recommender_System_Model as model
import GAP_Recommender_System_Utilities as gp

dataset = {"cc_id":[], "embeddings":[], "ranking":[]}
for _ in range(1000):
	embedding=np.random.choice(100,10)
	rank = np.random.choice(100,1).astype('float64')
	cc_id = np.random.choice(1000,1)
	dataset["cc_id"].append(str(cc_id[0]))
	dataset["embeddings"].append(embedding)
	dataset["ranking"].append(rank)
tf.debugging.experimental.enable_dump_debug_info(
    "/Users/colinfritz/Desktop/my_repos/GAP_Recommender_System_MVP/mvp_logs",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
print(len(set(dataset['cc_id'])))
dataset = tf.data.Dataset.from_tensor_slices(dataset)

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
