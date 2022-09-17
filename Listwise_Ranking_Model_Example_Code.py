import pprint
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})
movies = movies.map(lambda x: x["movie_title"])

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))


tf.random.set_seed(42)

# Split between train and tests sets, as before.
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# We sample 50 lists for each user for the training data. For each list we
# sample 5 movies from the movies the user rated.
train = tfrs.examples.movielens.sample_listwise(
    train,
    num_list_per_user=50,
    num_examples_per_list=5,
    seed=42
)
test = tfrs.examples.movielens.sample_listwise(
    test,
    num_list_per_user=1,
    num_examples_per_list=5,
    seed=42
)
class RankingModel(tfrs.Model):

  def __init__(self, loss):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids),
      tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 2, embedding_dimension)
    ])

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
    # We first convert the id features into embeddings.
    # User embeddings are a [batch_size, embedding_dim] tensor.
    user_embeddings = self.user_embeddings(features["user_id"])

    # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
    # tensor.
    movie_embeddings = self.movie_embeddings(features["movie_title"])

    # We want to concatenate user embeddings with movie emebeddings to pass
    # them into the ranking model. To do so, we need to reshape the user
    # embeddings to match the shape of movie embeddings.
    list_length = features["movie_title"].shape[1]
    user_embedding_repeated = tf.repeat(
        tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

    # Once reshaped, we concatenate and pass into the dense layers to generate
    # predictions.
    concatenated_embeddings = tf.concat(
        [user_embedding_repeated, movie_embeddings], 2)

    return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("user_rating")

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )


epochs = 30

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
mse_model = RankingModel(tf.keras.losses.MeanSquaredError())
mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
mse_model.fit(cached_train, epochs=epochs, verbose=False)

