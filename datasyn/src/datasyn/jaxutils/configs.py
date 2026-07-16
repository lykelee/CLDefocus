import os
from typing import Optional

import jax


def setup(
    enable_x64: Optional[bool] = None,
    disable_jit: Optional[bool] = None,
    debug_nans: Optional[bool] = None,
    xla_python_client_allocator: Optional[bool] = None,
):
    if enable_x64 is not None:
        jax.config.update("jax_enable_x64", enable_x64)
    if disable_jit is not None:
        jax.config.update("jax_disable_jit", disable_jit)
    if debug_nans is not None:
        jax.config.update("jax_debug_nans", debug_nans)
    if xla_python_client_allocator is not None:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = str(xla_python_client_allocator)


def set_preallocation(val: bool | float | None = None):
    if isinstance(val, bool):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(val)
    elif isinstance(val, float):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(val > 0.0)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(val)
        os.environ
    elif val is not None:
        raise ValueError()


def setup_cache(
    compilation_cache_dir="/tmp/jax_cache",
    persistent_cache_min_entry_size_bytes=-1,
    persistent_cache_min_compile_time_secs=0,
    persistent_cache_enable_xla_caches="xla_gpu_per_fusion_autotune_cache_dir",
):
    jax.config.update("jax_compilation_cache_dir", compilation_cache_dir)
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        persistent_cache_min_entry_size_bytes,
    )
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        persistent_cache_min_compile_time_secs,
    )
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches", persistent_cache_enable_xla_caches
    )


def easy_optics_setup():
    """
    We need x64 for precise optical computations because:
    - Ray tracing is often imprecise with float32.
    - CZT for PSF computation may work incorrectly with float32 (due to chirp factors).
    """
    setup_cache()
    setup(enable_x64=True)


def easy_synthesis_setup():
    """
    float32 is usually sufficient for synthesis without optical simulations (e.g., image convolution).
    """
    setup_cache()
    setup(enable_x64=False)
