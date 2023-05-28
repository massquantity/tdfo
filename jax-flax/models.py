from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn


class TwoTower(nn.Module):
    size_map: Dict[str, int]
    embed_dim: int
    init_fn: Callable = jax.nn.initializers.glorot_uniform()
    activation: Callable = jax.nn.swish
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.user_embed = nn.Embed(
            self.size_map["user"],
            self.embed_dim,
            dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.item_embed = nn.Embed(
            self.size_map["item"],
            self.embed_dim,
            dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.language_embed = nn.Embed(
            self.size_map["language"],
            self.embed_dim,
            dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.is_ebook_embed = nn.Embed(
            self.size_map["is_ebook"],
            self.embed_dim,
            dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.format_embed = nn.Embed(
            self.size_map["format"],
            self.embed_dim,
            dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.publisher_embed = nn.Embed(
            self.size_map["publisher"],
            self.embed_dim, dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.pub_decade_embed = nn.Embed(
            self.size_map["pub_decade"],
            self.embed_dim,
            dtype=self.dtype,
            embedding_init=self.init_fn,
        )
        self.user_fc1 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=self.init_fn
        )
        self.user_fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=self.init_fn
        )
        self.item_fc1 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=self.init_fn
        )
        self.item_fc2 = nn.Dense(
            self.embed_dim, dtype=self.dtype, kernel_init=self.init_fn
        )

    def __call__(self, x):
        user_embeds = self.get_user_embeddings(x)
        item_embeds = self.get_item_embeddings(x)
        # return (user_embeds * item_embeds).sum(axis=1)
        return jnp.einsum("ij,ij->i", user_embeds, item_embeds)

    def get_user_embeddings(self, x):
        user_embeds = self.user_embed(x["user_id"])
        return self.user_fc2(self.activation(self.user_fc1(user_embeds)))

    def get_item_embeddings(self, x):
        item_embeds = self.item_embed(x["item_id"])
        language_embeds = self.language_embed(x["language"])
        is_ebook_embeds = self.is_ebook_embed(x["is_ebook"])
        format_embeds = self.format_embed(x["format"])
        publisher_embeds = self.publisher_embed(x["publisher"])
        pub_decade_embeds = self.pub_decade_embed(x["pub_decade"])
        item_repr = jnp.concatenate(
            [
                item_embeds,
                language_embeds,
                is_ebook_embeds,
                format_embeds,
                publisher_embeds,
                pub_decade_embeds,
                jnp.expand_dims(x["avg_rating"], axis=1),
                jnp.expand_dims(x["num_pages"], axis=1),
            ],
            axis=1,
        )
        return self.item_fc2(self.activation(self.item_fc1(item_repr)))


def init_model(
    rng: jax.random.PRNGKey,
    size_map: dict,
    embed_dim: int,
    mixed_precision: bool = False,
):
    init_inputs = {
        "user_id": jnp.ones(1, dtype=jnp.int32),
        "item_id": jnp.ones(1, dtype=jnp.int32),
        "language": jnp.ones(1, dtype=jnp.int32),
        "is_ebook": jnp.ones(1, dtype=jnp.int32),
        "format": jnp.ones(1, dtype=jnp.int32),
        "publisher": jnp.ones(1, dtype=jnp.int32),
        "pub_decade": jnp.ones(1, dtype=jnp.int32),
        "avg_rating": jnp.ones(1, dtype=jnp.float32),
        "num_pages": jnp.ones(1, dtype=jnp.float32),
    }
    compute_dtype = get_dtype(mixed_precision)
    model = TwoTower(size_map, embed_dim, dtype=compute_dtype)
    params = model.init(rng, init_inputs)["params"]
    return model, params


def get_dtype(mixed_precision: bool):
    platform = jax.local_devices()[0].platform
    if mixed_precision:
        if platform == "tpu":
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float16
    else:
        dtype = jnp.float32
    return dtype


def _visualize_model_layers(model: nn.Module, inputs: Any):
    print(model.tabulate(jax.random.PRNGKey(0), inputs))
