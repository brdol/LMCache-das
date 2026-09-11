# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from functools import wraps

import torch

from lmcache.logging import init_logger
from lmcache.v1.memory_management import GPUMemoryAllocator

from lmcache_hcu.v1.hipfile import HIPFile

logger = init_logger(__name__)

_PATCHED_PARENT_ALLOCATOR_SEMANTICS = False


def _attach_parent_allocator(memory_obj, parent_allocator):
    if memory_obj is not None and hasattr(memory_obj, "parent_allocator"):
        memory_obj.parent_allocator = parent_allocator
    return memory_obj


def _attach_parent_allocator_batch(memory_objs, parent_allocator):
    if memory_objs is None:
        return None
    for memory_obj in memory_objs:
        _attach_parent_allocator(memory_obj, parent_allocator)
    return memory_objs


def patch_parent_allocator_semantics() -> None:
    """Patch upstream allocator parent ownership without copying memory_management.py.

    LMCache_merge wires parent_allocator through AddressManager,
    TensorMemoryAllocator, and GPUMemoryAllocator so MemoryObj.release/free paths go
    back to the outer allocator. Keep that behavior as method-level monkey patches
    instead of carrying a forked copy of upstream memory_management.py.
    """
    global _PATCHED_PARENT_ALLOCATOR_SEMANTICS
    if _PATCHED_PARENT_ALLOCATOR_SEMANTICS:
        return

    import lmcache.v1.memory_management as mm

    original_address_manager_init = mm.AddressManager.__init__

    @wraps(original_address_manager_init)
    def address_manager_init(self, size, align_bytes=mm.AddressManager.ALIGN_BYTES, parent_allocator=None):
        original_address_manager_init(self, size, align_bytes)
        self.parent_allocator = parent_allocator if parent_allocator is not None else self

    mm.AddressManager.__init__ = address_manager_init

    original_tensor_allocator_init = mm.TensorMemoryAllocator.__init__

    @wraps(original_tensor_allocator_init)
    def tensor_allocator_init(
        self,
        tensor,
        align_bytes=mm.AddressManager.ALIGN_BYTES,
        init_address_space=None,
        parent_allocator=None,
    ):
        original_tensor_allocator_init(
            self,
            tensor,
            align_bytes=align_bytes,
            init_address_space=init_address_space,
        )
        self.parent_allocator = parent_allocator if parent_allocator is not None else self
        self.address_manager = mm.AddressManager(
            self.buffer.numel() if init_address_space is None else init_address_space,
            align_bytes,
            parent_allocator=self.parent_allocator,
        )

    mm.TensorMemoryAllocator.__init__ = tensor_allocator_init

    original_tensor_allocate = mm.TensorMemoryAllocator.allocate

    @wraps(original_tensor_allocate)
    def tensor_allocate(self, *args, **kwargs):
        memory_obj = original_tensor_allocate(self, *args, **kwargs)
        return _attach_parent_allocator(
            memory_obj,
            getattr(self, "parent_allocator", self),
        )

    mm.TensorMemoryAllocator.allocate = tensor_allocate

    original_tensor_batched_allocate = mm.TensorMemoryAllocator.batched_allocate

    @wraps(original_tensor_batched_allocate)
    def tensor_batched_allocate(self, *args, **kwargs):
        memory_objs = original_tensor_batched_allocate(self, *args, **kwargs)
        return _attach_parent_allocator_batch(
            memory_objs,
            getattr(self, "parent_allocator", self),
        )

    mm.TensorMemoryAllocator.batched_allocate = tensor_batched_allocate

    # NOTE: GPUMemoryAllocator.__init__ is intentionally NOT patched here to
    # inject parent_allocator=self. GL 0.4.2-dcu sets MemoryObj.parent_allocator
    # to the inner TensorMemoryAllocator (inside TensorMemoryAllocator.allocate),
    # so MemoryObj.__del__ -> free lands on TensorMemoryAllocator.free (address
    # manager only -- no device_mem_lock). Injecting parent_allocator=self here
    # (=GPUMemoryAllocator) routed __del__ -> GPUMemoryAllocator.free, which
    # re-acquires the non-reentrant device_mem_lock while allocate already holds
    # it (GC fires __del__ mid-allocate once the 8 GiB XDS pool fills under a long
    # chunked prefill), deadlocking both TP workers. Leaving __init__ as baseline
    # keeps parent_allocator on the inner TensorMemoryAllocator (via
    # tensor_allocator_init's `parent_allocator if not None else self` fallback),
    # matching GL.

    _PATCHED_PARENT_ALLOCATOR_SEMANTICS = True


class HyFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, use_mla: bool = False, device=None):
        patch_parent_allocator_semantics()
        HIPFile.init_driver()

        if device is None:
            device = f"cuda:{torch.cuda.current_device()}"

        super().__init__(size, device, align_bytes=4096)
        self.base_pointer = self.tensor.data_ptr()
        HIPFile.register_global_buffer(self.base_pointer, size)
        logger.info(
            "Registering HyFile buffer at address %s with size %s",
            self.base_pointer,
            size,
        )

    def __del__(self):
        try:
            HIPFile.deregister_global_buffer(self.base_pointer)
            HIPFile.close_driver()
        except Exception:
            pass

    def __str__(self):
        return "HyFileMemoryAllocator"
