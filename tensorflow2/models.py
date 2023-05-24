import tensorflow as tf


class TwoTower(tf.keras.Model):

    def __init__(self, size_map, embed_dim):
        super().__init__()
        self.user_embed = tf.keras.layers.Embedding(
            size_map["user"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.item_embed = tf.keras.layers.Embedding(
            size_map["item"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.language_embed = tf.keras.layers.Embedding(
            size_map["language"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.is_ebook_embed = tf.keras.layers.Embedding(
            size_map["is_ebook"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.format_embed = tf.keras.layers.Embedding(
            size_map["format"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.publisher_embed = tf.keras.layers.Embedding(
            size_map["publisher"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.pub_decade_embed = tf.keras.layers.Embedding(
            size_map["pub_decade"], embed_dim, embeddings_initializer="glorot_uniform"
        )
        self.user_fc1 = tf.keras.layers.Dense(
            embed_dim, activation=tf.nn.swish, kernel_initializer="lecun_normal"
        )
        self.user_fc2 = tf.keras.layers.Dense(
            embed_dim, activation=None, kernel_initializer="lecun_normal"
        )
        self.item_fc1 = tf.keras.layers.Dense(
            embed_dim, activation=tf.nn.swish, kernel_initializer="lecun_normal"
        )
        self.item_fc2 = tf.keras.layers.Dense(
            embed_dim, activation=None, kernel_initializer="lecun_normal"
        )

    def call(self, inputs, **kwargs):
        user_embeds = self.get_user_embeddings(inputs)
        item_embeds = self.get_item_embeddings(inputs)
        return tf.reduce_sum(user_embeds * item_embeds, axis=1)
        # return tf.einsum("ij,ij->i", user_embeds, item_embeds)

    def get_user_embeddings(self, inputs):
        user_embeds = self.user_embed(inputs["user_id"])
        return self.user_fc2(self.user_fc1(user_embeds))

    def get_item_embeddings(self, inputs):
        item_embeds = self.item_embed(inputs["item_id"])
        language_embeds = self.language_embed(inputs["language"])
        is_ebook_embeds = self.is_ebook_embed(inputs["is_ebook"])
        format_embeds = self.format_embed(inputs["format"])
        publisher_embeds = self.publisher_embed(inputs["publisher"])
        pub_decade_embeds = self.pub_decade_embed(inputs["pub_decade"])
        item_repr = tf.keras.layers.Concatenate(axis=1)(
            [
                item_embeds,
                language_embeds,
                is_ebook_embeds,
                format_embeds,
                publisher_embeds,
                pub_decade_embeds,
                tf.expand_dims(inputs["avg_rating"], axis=1),
                tf.expand_dims(inputs["num_pages"], axis=1),
            ],
        )
        return self.item_fc2(self.item_fc1(item_repr))
