from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from datasyn.jaxutils.typing import *

UInt32 = jnp.uint32


def str_to_u32(x: str) -> np.uint32:
    import hashlib

    data = x.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return np.frombuffer(digest[:4], dtype=np.uint32)[0]


def tag_seed(seed: int, tag: str) -> int:
    """
    "Tags" an integer seed with a string, deriving a new seed.

    Useful to derive independent RNG streams from one global seed by tagging
    with task names (e.g., "generation" vs "evaluation").

    Mixing is done by hashing ``f"{seed}/{tag}"`` with SHA-256.
    """
    import hashlib

    data = f"{int(seed)}/{tag}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "little")


HRngToken = str | IntArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HRng:
    """
    Hierarchical RNG for JAX based on `jax.random.fold_in`.

    - `HRng.from_seed(seed)` creates a root RNG.
    - `rng[tok]` derives a child RNG by folding in `tok`.
      * `tok` may be a string (hashed to uint32) or an int/array (cast to uint32).
    - `split(n)` optionally introduces a leading batch dimension on the key.
    """

    _key: RngKey

    @staticmethod
    def from_seed(seed: int | jax.Array) -> "HRng":
        seed_u32 = jnp.asarray(seed, UInt32)
        return HRng(jax.random.PRNGKey(seed_u32))

    @staticmethod
    def from_key(key: RngKey) -> "HRng":
        return HRng(_key=key)

    @property
    def key(self) -> RngKey:
        return self._key

    @property
    def device(self) -> JaxDevice:
        return self._key.device

    def __getitem__(self, tok: HRngToken) -> "HRng":
        if isinstance(tok, tuple):
            rng = self
            for t in tok:
                rng = rng[t]
            return rng
        if isinstance(tok, str):
            tok = str_to_u32(tok)
        newkey = jax.random.fold_in(
            self._key, jnp.asarray(tok, UInt32, device=self.device)
        )
        return HRng.from_key(newkey)

    def key_of(self, tok: HRngToken) -> RngKey:
        return self[tok].key

    def split(self, n: int = 2) -> tuple["HRng", ...]:
        keys = jax.random.split(self._key, n)
        return tuple(HRng.from_key(k) for k in keys)

    def split_asbatch(self, n: int = 2) -> "HRng":
        keys = jax.random.split(self._key, n)
        return HRng.from_key(keys)
