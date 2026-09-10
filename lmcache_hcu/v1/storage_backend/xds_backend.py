# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote
import asyncio
import ctypes
import os
import random
import string
import struct
import tempfile
import threading
import time
import mmap
import numpy as np

# Third Party
import torch
from abc import ABC, abstractmethod


# First Party
# from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache_hcu.v1.storage_backend.xds_metadata import XDSCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    HyFileMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache_hcu.v1.hipfile import HIPFile
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvdata"
_METADATA_FILE_SUFFIX_TMP = ".METADATA"
_DATA_FILE_SUFFIX_TMP = ".KVDATA"
_METADATA_VERSION = 1

# gds、 weka均为4096，但是是填充的，此处72为经验值
_METADATA_MAX_SIZE = 72
_ALIGNMENT = 4096  # 4K 对齐
_DIR_SIZE = 2
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2


def _close_file_descriptor(fd):
    """Close a fallback file descriptor without masking the I/O failure."""
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


class UnsupportedMetadataVersion(Exception):
    pass


torch_dtypes = [
    torch.half,
    torch.float16,
    torch.bfloat16,
    torch.float,
    torch.float32,
    torch.float64,
    torch.double,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
dtype_to_idx = {dtype: idx for idx, dtype in enumerate(torch_dtypes)}


def _verify_hyfile_roundtrip(
    xds,
    allocator,
    root_path,
):
    """Fail closed unless this allocator passes one real HyFile roundtrip."""
    required_size = 2 * _ALIGNMENT
    if allocator.tensor.numel() < required_size:
        raise RuntimeError(
            "HyFile preflight requires at least an 8 KiB registered buffer"
        )
    device_offset = _ALIGNMENT
    test_buffer = allocator.tensor[device_offset:required_size]

    fd, path = tempfile.mkstemp(prefix=".lmcache-xds-preflight-", dir=root_path)
    os.close(fd)
    try:
        test_buffer.fill_(0xA5)
        if test_buffer.is_cuda:
            torch.cuda.synchronize(test_buffer.device)

        size = test_buffer.numel()
        ret = xds.write(path, allocator.base_pointer, size, 0, device_offset)
        if ret != size:
            raise RuntimeError(f"HyFile preflight write incomplete: {ret}/{size}")

        expected = bytes([0xA5]) * size
        with open(path, "rb") as f:
            actual = f.read()
        if actual != expected:
            raise RuntimeError(
                "HyFile preflight write readback mismatch; refusing unsafe XDS startup"
            )

        test_buffer.zero_()
        if test_buffer.is_cuda:
            torch.cuda.synchronize(test_buffer.device)
        ret = xds.read(path, allocator.base_pointer, size, 0, device_offset)
        if ret != size:
            raise RuntimeError(f"HyFile preflight read incomplete: {ret}/{size}")
        if test_buffer.is_cuda:
            torch.cuda.synchronize(test_buffer.device)
        if test_buffer.detach().cpu().numpy().tobytes() != expected:
            raise RuntimeError(
                "HyFile preflight DMA readback mismatch; refusing unsafe XDS startup"
            )
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class XdsWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:

        if task_type == "prefetch":
            priority = 0
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        try:
            result = await self.executor.submit_job(
                task,
                *args,
                priority=priority,
                **kwargs,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to submit job {task_type}: {e}", exc_info=True)
            raise

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


def pack_metadata(shape, dtype, size) -> bytes:
    metadata_desc = "<QQQQ" + len(shape) * "Q"
    if struct.calcsize(metadata_desc) > _METADATA_MAX_SIZE:
        # TODO(Serapheim/Ilya): support variable offset for data
        raise ValueError(
            f"Metadata size {struct.calcsize(metadata_desc)} "
            f"exceeds max size {_METADATA_MAX_SIZE}"
        )
    return struct.pack(
        metadata_desc, _METADATA_VERSION,
        dtype_to_idx[dtype],
        size,
        len(shape),
        *shape
    )


def unpack_metadata(buffer):
    # A truncated/corrupt sidecar must surface as UnsupportedMetadataVersion:
    # the startup scan only catches that type, so a bare struct.error here
    # would kill the whole scan task on one bad file (typical crash leftover).
    try:
        version, dt_idx, size, ndim = struct.unpack_from("<QQQQ", buffer)
        if version != 1:
            raise UnsupportedMetadataVersion(
                f"Unsupported metadata version: {version}"
            )
        if dt_idx >= len(torch_dtypes):
            raise UnsupportedMetadataVersion(f"bad dtype code {dt_idx}")
        shape_offset = struct.calcsize("<QQQQ")
        shape = struct.unpack_from("<" + ndim * "Q", buffer, offset=shape_offset)
        return torch.Size(shape), torch_dtypes[dt_idx], size
    except struct.error as e:
        raise UnsupportedMetadataVersion(f"corrupt metadata: {e}") from e


def rand_suffix(rand, n: int):
    return "".join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP) + tmp
    with open(tmp_path, "wb") as f:
        f.write(metadata)
    os.rename(tmp_path, path + _METADATA_FILE_SUFFIX)


class CliStorageInterface(ABC):

    @abstractmethod
    def initialize(self) -> bool:
        """初始化 XDS 驱动，返回是否成功"""
        pass

    @abstractmethod
    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        """
        写入数据到文件
        返回：实际写入的字节数，失败返回负数
        """
        pass

    @abstractmethod
    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:
        """
        从文件读取数据
        返回：实际读取的字节数，失败返回负数
        """
        pass

    def cleanup(self):
        pass


class HyFileDDS(CliStorageInterface):
    """Hygon DCU HyFile DDS 实现"""

    # hipfile DMA transfers are NOT thread-safe: concurrent hipFileWrite calls
    # from the put-executor workers corrupt each other's payloads (observed
    # e2e as scattered K-region corruption in .kvdata; single-writer run is
    # byte-exact). Serialize all driver I/O — the DMA engine is serial anyway,
    # so the lock costs little.
    _driver_lock = threading.Lock()

    def initialize(self) -> bool:
        try:
            self.hiprt = ctypes.CDLL("libhiprtc.so")
            logger.info("HyFile DDS initialized")
            return True
        except Exception as e:
            # libhiprtc.so is ONLY used for the CPU-copy fallback when a GPUDirect
            # write/read is incomplete; the primary GPUDirect path does not need
            # it. So do NOT fail here (returning False makes create_xds_interface
            # raise, which would block an otherwise-working GPUDirect-only host).
            # If the fallback is ever needed with hiprt=None, write()/read() return
            # -1 loudly and the store fails there -- no silent corruption. The old
            # `f"HIP Runtime Error"` dropped the real exception e.
            logger.warning(
                f"libhiprtc.so unavailable ({e}); XDS CPU-copy fallback disabled, "
                "primary GPUDirect path unaffected."
            )
            self.hiprt = None
            return True

    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        # 尝试通过XDS写入
        try:
            with self._driver_lock:
                with HIPFile(file_path, "w", skip_buffer_registration=True) as f:
                    ret = f.write(gpu_address, size, file_offset, dev_offset)
            if ret == size:
                logger.debug(f"HyFile write: {size} bytes")
                return size
            logger.warning(
                f"hyFile write incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )
        except Exception as e:
            logger.warning(f"hyFile write failed: {e}, falling back to CPU copy")

        # XDS写入失败后回退至POSIX写入
        if self.hiprt is None:
            logger.error("HIP runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None

        # 尝试通过 HIP Runtim + mmap 写入
        try:
            fd = os.open(file_path, os.O_RDWR)
            os.ftruncate(fd, file_offset + size)
            # mmap 文件
            mm = mmap.mmap(
                fd, file_offset + size,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED
            )
            os.close(fd)
            fd = None

            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 hipMemcpy 从 GPU 拷贝到 CPU
            res = self.hiprt.hipMemcpy(
                ctypes.c_void_p(cpu_addr + file_offset),        # dst: CPU 内存
                ctypes.c_void_p(gpu_address + dev_offset),      # src: GPU 内存
                ctypes.c_size_t(size),                          # size
                ctypes.c_int(MEMCPY_DEVICE_TO_HOST),  # direction: D2H
            )

            if res != 0:
                logger.error(f"Memcpy failed with code {res}")
                return -1

            logger.info(f"CPU fallback write successful: {size} bytes")
            return size
        except Exception as e:
            logger.error(f"CPU fallback write failed: {e}", exc_info=True)
            return -1

        finally:
            if arr is not None:
                del arr
            if mm is not None:
                mm.close()
            _close_file_descriptor(fd)

    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:

        if not os.path.exists(file_path):
            logger.error(f"No such file or directory: {file_path}, read failed")
            return -1

        # 尝试通过XDS读取
        try:
            with self._driver_lock:
                with HIPFile(file_path, "r", skip_buffer_registration=True) as f:
                    ret = f.read(gpu_address, size, file_offset, dev_offset)
                    if ret == size:
                        return ret
            logger.warning(
                f"hyFile read incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )
        except Exception as e:
            logger.warning(f"hyFile read failed: {e}, falling back to CPU copy")

        # XDS读取失败后回退至POSIX写入
        if self.hiprt is None:
            logger.error("HIP runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None

        # 尝试通过 HIP Runtim + mmap 读取
        try:
            # 打开文件并 mmap
            fd = os.open(file_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd, file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)
            fd = None

            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 hipMemcpy 拷贝到 GPU
            res = self.hiprt.hipMemcpy(
                ctypes.c_void_p(gpu_address + dev_offset),
                ctypes.c_void_p(cpu_addr + file_offset),
                ctypes.c_size_t(size),
                ctypes.c_int(MEMCPY_HOST_TO_DEVICE),
            )

            if res != 0:
                logger.error(f"hipMemcpy failed with code {res}")
                return -1

            logger.info(f"CPU fallback successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return -1

        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()
            _close_file_descriptor(fd)


class Posix(CliStorageInterface):

    def __init__(self, root_path):
        self.use_odirect = True

        # Block size (for file system I/O)
        stat = os.statvfs(root_path)
        self.os_disk_bs = stat.f_bsize

    def initialize(self) -> bool:
        return True

    def write(self, path:str, kv_chunk:torch.Tensor, size:int,  buffer):
        assert len(buffer) == size
        tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP)

        metadata = pack_metadata(
            kv_chunk.shape,
            kv_chunk.dtype,
            size
        )

        start_time = time.time()

        try:
            if size % self.os_disk_bs != 0 or not self.use_odirect:
                with open(tmp_path, "wb") as f:
                    written = f.write(buffer)
            else:
                fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                try:
                    written = os.write(fd, buffer)
                finally:
                    os.close(fd)
            if written != size:
                raise OSError(f"POSIX write incomplete: {written}/{size} bytes")

            os.rename(tmp_path, path)

            disk_write_time = time.time() - start_time
            logger.debug(
                f"clistorage write size: {size} bytes, "
                f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
            )
        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise e
        return metadata

    def read(self, buffer, path):
        start_time = time.time()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to clistorage block size."
            )

        try:
            if not fblock_aligned or not self.use_odirect:
                with open(path, "rb") as f:
                    bytes_read = f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    bytes_read = fdo.readinto(buffer)
        except FileNotFoundError:
            logger.warning(f"File not found on clistorage: {path}")
            return -1

        disk_read_time = time.time() - start_time
        logger.debug(
            f"clistorage read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )
        # Return the ACTUAL bytes read, not the requested size. A short read
        # (truncated/partial .data file) then surfaces as ret != size at the
        # caller (_load_bytes_from_storage_with_memory) and is treated as a miss
        # instead of silently accepting a buffer with a stale/garbage tail.
        return bytes_read


class CuFileGDS(CliStorageInterface):
    """NVIDIA cuFile GDS 实现"""
    def __init__(self):
        import cufile
        self.cufile = cufile
        self._driver = None

    def initialize(self) -> bool:
        try:
            self._driver = self.cufile.CuFileDriver()
            logger.info("cuFile GDS initialized")
            self.cudart = None
            self.cudart = ctypes.CDLL("libcudart.so")
            logger.info("CUDA runtime loaded for fallback")
            return True
        except Exception as e:
            logger.error(f"cuFile init error: {e}")
            return False

    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        try:
            # 尝试 cuFile 直接写入
            with self.cufile.CuFile(file_path, "r+", use_direct_io = True) as f:
                ret = f.write(
                    ctypes.c_void_p(gpu_address), size,
                    file_offset=file_offset, dev_offset=dev_offset
                )

            # 如果写入完整，直接返回
            if ret == size:
                return ret

            # 写入不完整，尝试回退
            logger.warning(
                f"cuFile write incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )

        except Exception as e:
            logger.warning(f"cuFile write failed: {e}, falling back to CPU copy")

        # ========== 回退到 CPU 拷贝 ==========
        if self.cudart is None:
            logger.error("CUDA runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None
        try:

            fd = os.open(file_path, os.O_RDWR)
            os.ftruncate(fd, file_offset + size)
            # mmap 文件
            mm = mmap.mmap(
                fd, file_offset + size,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED
            )
            os.close(fd)
            fd = None
            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 cudaMemcpy 从 GPU 拷贝到 CPU
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(cpu_addr + file_offset),        # dst: CPU 内存
                ctypes.c_void_p(gpu_address + dev_offset),      # src: GPU 内存
                ctypes.c_size_t(size),                          # size
                ctypes.c_int(MEMCPY_DEVICE_TO_HOST),  # direction: D2H
            )

            if res != 0:
                logger.error(f"cudaMemcpy failed with code {res}")
                return -1
            # 确保数据写入磁盘
            logger.info(f"CPU fallback write successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback write failed: {e}", exc_info=True)
            return -1
        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()
            _close_file_descriptor(fd)

    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:

        if not os.path.exists(file_path):
            logger.error(f"No such file or directory: {file_path}, read failed")
            return -1

        try:
            with self.cufile.CuFile(file_path, "r", use_direct_io = True) as f:
                ret = f.read(
                    ctypes.c_void_p(gpu_address), size,
                    file_offset=file_offset, dev_offset=dev_offset
                )

            if ret == size:
                return ret

            # 读取不完整，尝试回退
            logger.warning(
                f"cuFile read incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )

        except Exception as e:
            logger.warning(f"cuFile read failed: {e}, falling back to CPU copy")

        if self.cudart is None:
            logger.error("CUDA runtime not available, cannot fallback")
            return -1
        fd = None
        mm = None
        arr = None
        try:
            # 打开文件并 mmap
            fd = os.open(file_path, os.O_RDONLY)

            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd, file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)
            fd = None
            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 cudaMemcpy 拷贝到 GPU
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(gpu_address + dev_offset),
                ctypes.c_void_p(cpu_addr + file_offset),
                ctypes.c_size_t(size),
                ctypes.c_int(MEMCPY_HOST_TO_DEVICE),
            )

            if res != 0:
                logger.error(f"cudaMemcpy failed with code {res}")
                return -1
            logger.info(f"CPU fallback successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return -1
        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()
            _close_file_descriptor(fd)


def check_environment():
    if torch.version.cuda is None:
        backend = "hyfile"
    else:
        backend = "cufile"

    logger.info(f"Detected XDS interface: {backend}")

    return backend


def create_xds_interface(backend: str):
    # 创建实例
    if backend == "cufile":
        xds = CuFileGDS()
    else:
        xds = HyFileDDS()

    if not xds.initialize():
        raise RuntimeError(f"Failed to initialize {backend}")

    return xds


class XdsBackend(AllocatorBackendInterface):
    """
    This is a backend that leverages Hygon's HyFile API or NVIDIA's cuFile API and to issue XDS requests
    directly to the storage.  In order to use it, users need to specify
    `xds_path` , `xds_buffer_size` and `max_xds_size` in their LMCache config.

    Cache Directory Structure created by this Backend:
    /{xds_path}/{first_level}/{second_level}/{data & metadata}
    This structure is semi-arbitrary.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
        local_cpu_backend: LocalCPUBackend = None,
    ):
        assert os.path.exists(config.xds_path), (
            f"Xds path {config.xds_path} does not exist"
        )

        assert dst_device.startswith("cuda")
        super().__init__(dst_device=dst_device)

        self.config = config
        self.loop = loop
        if self.config.xds_buffer_size is not None:
            # 使用xds协议
            self.xds_interface = check_environment()
            self.memory_allocator = self.initialize_allocator(config, metadata, self.xds_interface)
            self.xds = create_xds_interface(self.xds_interface)

            if self.memory_allocator is not None:
                assert hasattr(self.memory_allocator, "base_pointer")
                self.base_pointer = self.memory_allocator.base_pointer
                if self.xds_interface == "hyfile":
                    # One real write+readback roundtrip before serving traffic:
                    # a driver that silently writes zeros would otherwise
                    # corrupt every cached entry (fail closed, not fail open).
                    try:
                        _verify_hyfile_roundtrip(
                            self.xds, self.memory_allocator, config.xds_path
                        )
                    except BaseException:
                        # Failed construction: release the driver and the GPU
                        # pool we just created, or they leak with the raise.
                        self.xds.cleanup()
                        self.memory_allocator.close()
                        raise

        else:
            # 使用posix协议
            assert local_cpu_backend is not None
            self.memory_allocator = local_cpu_backend
            self.xds = Posix(config.xds_path)

        self.dst_device = dst_device

        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT
        else:
            self.fmt = MemoryFormat.KV_2LTD

        self.xds_path = config.xds_path

        self.hot_lock = threading.Lock()
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

        self.metadata_dirs: set[str] = set()

        self.rand = random.Random(self.dst_device)

        thread_count = config.get_extra_config_value('xds_io_threads', default_value=4)

        logger.info(f"Using {thread_count} threads for XDS I/O")
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_count, thread_name_prefix="xds-io"
        )

        self.xds_worker = XdsWorker(self.loop)

        self.world_id = metadata.worker_id

        if metadata.use_mla:
            self.world_size = 1
        else:
            self.world_size = metadata.world_size

        if config.max_xds_size is not None:
            self.eviction = True
            self.max_xds_size = int(config.max_xds_size * 1024**3 )
            logger.info(f"Peer rank max_storage_size: {config.max_xds_size}GB, "
                        f"total max_storage_size: {config.max_xds_size * self.world_size}GB")
        else:
            self.eviction = False
            logger.info(f"config.max_xds_size is None, don't suppot eviction")

        # flag for extra assertions to catch bugs but harm performance
        self._debug_asserts = False
        # flag to use O_NOATIME during metadata file read for performance improvement
        self._use_noatime = True

        self.current_cache_size = 0.0
        self.keys_in_request: List[CacheEngineKey] = []

        self._scan_future = asyncio.run_coroutine_threadsafe(
            self._scan_metadata(), self.loop
        )

    async def _scan_metadata(self):
        """分批扫描元数据，带进度反馈"""
        BATCH_SIZE = 11  # 每批处理的目录数

        start = time.perf_counter()

        # 第一步：收集所有待扫描的目录
        dirs_to_scan = []
        with os.scandir(self.xds_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                if len(entry.name) != _DIR_SIZE:
                    continue
                dirs_to_scan.append({
                    'path': os.path.join(self.xds_path, entry.name),
                    'l1_dir': entry.name
                })

        total_dirs = len(dirs_to_scan)
        processed = 0
        errors = 0

        logger.info(f"Starting metadata scan: {total_dirs} directories to process")

        # 第二步：分批处理
        for i in range(0, total_dirs, BATCH_SIZE):
            batch = dirs_to_scan[i:i + BATCH_SIZE]

            # 创建当前批次的任务
            tasks = [
                asyncio.to_thread(
                    self._scan_metadata_subdir,
                    item['path'],
                    item['l1_dir']
                )
                for item in batch
            ]

            # 等待当前批次完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计错误
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    logger.error(f"Batch scan error: {result}")

            processed += len(batch)
            progress = (processed / total_dirs) * 100

            logger.info(
                f"Scan progress: {processed}/{total_dirs} ({progress:.1f}%), "
                f"errors: {errors}, Cache entries: {len(self.hot_cache)}"
            )

        end = time.perf_counter()
        # with self.hot_lock:
        #     self.current_cache_size /= self.world_size
        logger.info(
            f"Metadata scan complete: {len(self.hot_cache)} entries "
            f"in {end - start:.2f}s, peer rank size:{self.current_cache_size}, errors: {errors}"
        )

    def _scan_metadata_subdir(self, path, l1_dir):
        target_suffix = _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != _DIR_SIZE:
                    continue
                with os.scandir(os.path.join(path, l2_dir)) as it2:
                    for fentry in it2:
                        if not fentry.is_file():
                            continue
                        if not fentry.name.endswith(target_suffix):
                            # A live put may still own these tmp files (.KVDATA /
                            # .KVDATA.METADATA); skip instead of deleting, or a
                            # concurrent put's tmp file gets removed underneath it.
                            if fentry.name.endswith(
                                (_DATA_FILE_SUFFIX_TMP, _METADATA_FILE_SUFFIX_TMP)
                            ):
                                continue
                            if not fentry.name.endswith(_DATA_FILE_SUFFIX):
                                os.remove(fentry.path)
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = unquote(filename[: -len(target_suffix)])
                        try:
                            key = CacheEngineKey.from_string(key_str)
                        except ValueError as e:
                            logger.error(
                                f"Filename {filename} can't be converted "
                                f"back into cache key: {e}"
                            )
                            continue
                        # Value equality, not identity: `is not` only happens to
                        # work for CPython's small-int cache (-5..256); a
                        # worker_id >= 257 (or any non-interned int) would make
                        # `is not` spuriously true and skip a valid on-disk entry.
                        if key.worker_id != self.world_id:
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for "
                                f"{fentry.path}, ignoring"
                            )

    # 修改时间： 2026.03.27
    # 修改人  ： 郭丰雅
    # 修改原因： 合并官方gds性能提升的代码 https://github.com/LMCache/LMCache/pull/2637/changes
    def _read_metadata_info(self, filename: str):
        # Use O_NOATIME to prevent updating access time and improve performance
        # Instead of using Python's open() and read(), we use the OS's open() and
        # read() because it is faster - the metadata file is small and we don't
        # need any buffering.
        # Additionally, we use O_NOATIME to improve performance
        if self._use_noatime:
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOATIME)
            except (
                # PermissionError: User doesn't own the file
                # AttributeError: O_NOATIME not available on this platform
                # OSError: Filesystem doesn't support O_NOATIME (EINVAL)
                PermissionError,
                AttributeError,
                OSError,
            ):
                # fallback to normal open if O_NOATIME is not supported
                self._use_noatime = False
                logger.info(
                    "O_NOATIME flag not supported during metadata file read, "
                    "falling back to normal open"
                )
                fd = os.open(filename, os.O_RDONLY)
        else:
            fd = os.open(filename, os.O_RDONLY)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE)
        finally:
            os.close(fd)
        return unpack_metadata(buf)


    def _read_metadata(self, key, filename, subdir_key):

        shape, dtype, size = self._read_metadata_info(filename)

        # Reject metadata whose recorded size is inconsistent with shape/dtype,
        # else a corrupt size drives a wrong-length disk read (xds path: the DMA
        # overruns into neighboring entries). The posix allocator's alignment is
        # configurable, so only size >= raw is guaranteed; the xds allocator is
        # fixed at 4K alignment, so size must equal the aligned size exactly.
        raw = shape.numel() * dtype.itemsize
        if size < raw:
            raise UnsupportedMetadataVersion(
                f"metadata size {size} < raw payload {raw} for {filename}"
            )
        if self.config.xds_buffer_size is not None:
            expected = (raw + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            if size != expected:
                raise UnsupportedMetadataVersion(
                    f"metadata size {size} != aligned {expected} for {filename}"
                )

        metadata = XDSCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,      # 原始大小
            shape,
            dtype
        )

        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
            self.current_cache_size += metadata.size

        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO(Serapheim): implement pin() semantics
        with self.hot_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                self.keys_in_request.append(key)
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.hot_lock:
            for key in keys:
                if key not in self.hot_cache:
                    return num_hit_counts
                if pin:
                    self.hot_cache[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        logger.info(f"batched_async_contains: {num_hit_counts}")
        return num_hit_counts

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[XDSCacheMetadata]:
        path, subdir_key, _, _ = self._key_to_path(key)
        path += _METADATA_FILE_SUFFIX
        if os.path.exists(path):
            try:
                return self._read_metadata(key, path, subdir_key)
            except UnsupportedMetadataVersion:
                logger.error(f"Unsupported metadata version for {path}, ignoring")
        return None

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> Tuple[str, str, str, str]:
        hash_str = str(key.chunk_hash)
        l1_dir = hash_str[:_DIR_SIZE]
        l2_dir = hash_str[_DIR_SIZE:_DIR_SIZE*2]
        # Encode the key into a flat, reversible filename via urllib quote
        # (safe=""). The old scheme did key_str.replace("/", "_") and asserted
        # the key had no "_", which rejected any model whose key string contains
        # "_" (e.g. model names with underscores -> AssertionError, so XDS store
        # raised before writing any file). quote/unquote round-trips "_", "/"
        # and every other character. Decoded back via unquote() in the scan.
        key_str = key.to_string()
        path = os.path.join(
                self.xds_path,
                l1_dir,
                l2_dir,
                quote(key_str, safe="") + _DATA_FILE_SUFFIX
            )
        return (path, l1_dir + l2_dir, l1_dir, l2_dir)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.xds_worker.exists_in_put_tasks(key)

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        # This backend fires the put as a coroutine and always returns None (like
        # local_disk_backend); the annotation is Optional[Future] to match reality
        # rather than the old bare `-> Future` that never held.
        if self.memory_allocator is None:  # MLA non-first-rank: inert, no offload
            return None
        assert memory_obj.tensor is not None

        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.xds_worker.insert_put_task(key)

        # Accounting is unified on the *data* physical size only (matches the
        # official local_disk_backend `required_size = get_physical_size()` and
        # the per-entry `.size` stored in insert_key / used at load-time
        # `ret != size`). The metadata sidecar (~<= _METADATA_MAX_SIZE bytes) is
        # NOT added here: it was double-counted before (add included +512 while
        # every subtract used `.size` == phys), leaking _METADATA_MAX_SIZE per
        # evicted key and drifting current_cache_size above max_xds_size — a leak
        # the Step-4 72->512 bump amplified 7x. Keep a single, consistent unit.
        required_size = memory_obj.get_physical_size()
        evict_success = True
        if self.eviction:
            evict_success = self._eviction(required_size)
        else:
            self.current_cache_size += required_size

        if not evict_success:
            self.xds_worker.remove_put_task(key)
            return None

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.xds_worker.submit_task(
                "put",
                self._async_save_bytes_to_storage,
                key=key,
                memory_obj=memory_obj,
            ),
            self.loop,
        )

    def _eviction(self, required_size: int) -> bool:
        # Two-phase eviction (fixes the duplicate-eviction bug, ADAPTATION_NOTES
        # §4): the memory index (hot_cache pop + counter + policy) is updated
        # INSIDE the lock so each candidate actually advances the LRU cursor, and
        # the slow file unlink is done OUTSIDE the lock. We must NOT call
        # self.remove()/batched_remove() here — remove() re-acquires the
        # non-reentrant hot_lock (deadlock) and would double-decrement the
        # counter for keys already accounted below.
        evicted_paths: list[str] = []
        evict_success = True
        with self.hot_lock:
            while self.current_cache_size + required_size > self.max_xds_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.hot_cache, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    # pop INSIDE the loop so get_evict_candidates advances to the
                    # next-oldest key next iteration instead of returning this
                    # same key forever (the original bug: index update lived
                    # outside the loop, so >=2-key evictions looped on one key).
                    meta = self.hot_cache.pop(evict_key)
                    self.cache_policy.update_on_force_evict(evict_key)
                    self.current_cache_size -= meta.size
                    evicted_paths.append(meta.path)

            if evict_success:
                self.current_cache_size += required_size
        # Slow remote-storage unlink outside the lock (satisfies the original
        # TODO). Each file is removed independently so a missing sidecar does not
        # orphan the data blob (or vice versa).
        for path in evicted_paths:
            for p in (path + _METADATA_FILE_SUFFIX, path):
                try:
                    os.remove(p)
                except OSError:
                    pass

        return evict_success

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def _async_save_bytes_to_storage(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to storage.
        """
        # logger.info(f"_async_save_bytes_to_storage {key}")
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None

        size = memory_obj.get_physical_size()
        path, _, l1_dir, l2_dir = self._key_to_path(key)

        os.makedirs(os.path.join(self.xds_path, l1_dir, l2_dir), exist_ok=True)

        try:
            if self.config.xds_buffer_size is not None:
                # xds 写入
                metadata = self._save_xds(
                        path,
                        kv_chunk,
                        size,
                        self.base_pointer,
                        memory_obj.metadata.address)
            else:
                # posix 写入
                buffer = memory_obj.byte_array
                metadata = self._save_posix(
                        path,
                        kv_chunk,
                        size,
                        buffer)

            save_metadata(path, _METADATA_FILE_SUFFIX_TMP, metadata)

            # Publish the in-memory index only after both data and metadata are
            # committed; readers/evictors must never see a half-committed entry
            # (visible in hot_cache => both files are on disk).
            self.insert_key(key, memory_obj)

        except Exception as e:
            logger.error(f"Failed to save {key}: {e}", exc_info=True)
            cleanup_paths = [
                path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP), # DATA的临时路径
                path, # DATA的正式路径
                path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP) + _METADATA_FILE_SUFFIX_TMP, # metadata的临时路径
                path + _METADATA_FILE_SUFFIX # metadata的正式路径
                ]
            for cleanup_path in cleanup_paths:
                if os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass
            with self.hot_lock:
                self.hot_cache.pop(key,None)
                # Undo exactly the reservation made in submit_put_task
                # (required_size == get_physical_size(), no metadata term).
                self.current_cache_size -= memory_obj.get_physical_size()
        finally:
            memory_obj.ref_count_down()
            self.xds_worker.remove_put_task(key)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        with self.hot_lock:
            if key in self.hot_cache:
                self.cache_policy.update_on_hit(key,self.hot_cache)
            else:
                path, _, _, _ = self._key_to_path(key)
                size = memory_obj.get_physical_size()
                shape = memory_obj.metadata.shape
                dtype = memory_obj.metadata.dtype
                self.hot_cache[key] = XDSCacheMetadata(path,size, shape, dtype)
                # cache_size_mb = asizeof.asizeof(self.hot_cache) / 1024 / 1024
                # logger.info(f"Hot cache size: {cache_size_mb:.2f} MB ({len(self.hot_cache)} entries)")

    async def _async_load_bytes_from_storage(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_storage_with_allocation(key, path, dtype, shape)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return None
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None
        self.cache_policy.update_on_hit(key, self.hot_cache)
        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        assert dtype is not None
        assert shape is not None
        return self._load_bytes_from_storage_with_allocation(key, path, dtype, shape)

    def submit_prefetch_task(self, key: CacheEngineKey) -> bool:
        # TODO(Jiayi): Need to modify this when prefetch interface is determined.
        # Mirror GdsBackend: prefetch is not implemented yet, always returns False.
        return False

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # No async get path yet. Returning None lets StorageManager fall
        # through to the blocking get; a bare Future() here would never be
        # resolved and leave waiters hanging.
        return None

    def _load_bytes_from_storage_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """
        Load byte array from storage into a pre-allocated memory object.

        Args:
            key: Cache key for error handling
            path: File path to load from
            memory_obj: Pre-allocated memory object to load data into

        Returns:
            The memory object with loaded data, or None if loading failed
        """
        if memory_obj is None or memory_obj.tensor is None:
            return None
        if self.config.xds_buffer_size is not None:
            assert memory_obj.tensor.is_cuda
            assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        with self.hot_lock:
            entry = self.hot_cache.get(key)

        if entry is None:
            logger.error(f"Metadata not found for {key}")
            return None

        if self.config.xds_buffer_size is not None:
            # XDS direct path stores the 4K-aligned physical size on disk.
            size = entry.size
            ret = self._load_xds(
                path,
                file_offset=0,
                gpu_pointer=ctypes.c_void_p(self.base_pointer),
                size=size,
                dev_offset=memory_obj.metadata.address
            )
        else:
            # POSIX stores the raw payload; checking against the aligned
            # entry.size would misclassify every non-4K-multiple entry as a
            # short read and evict it.
            assert entry.shape is not None and entry.dtype is not None
            size = entry.shape.numel() * entry.dtype.itemsize
            buffer = memory_obj.byte_array
            ret = self._load_posix(buffer[:size], path)

        if ret != size:
            logger.error(f"Load failed for {key}: {ret}/{size}")
            with self.hot_lock:
                if (meta := self.hot_cache.pop(key, None)):
                    size = meta.size
                    self.current_cache_size -= size
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def _load_bytes_from_storage_with_allocation(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from storage by first allocating memory, then loading.

        Args:
            key: Cache key for error handling
            path: File path to load from
            dtype: Data type for memory allocation
            shape: Shape for memory allocation

        Returns:
            A new memory object with loaded data, or None if allocation or
            loading failed
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync storage load.")
            return None
        assert memory_obj.tensor is not None
        if self.config.xds_buffer_size is not None:
            # GPU residency only holds for the XDS direct buffer; the posix
            # branch allocates from the (CPU) LocalCPU backend.
            assert memory_obj.tensor.is_cuda
            assert torch.device(self.dst_device) == torch.device(
                memory_obj.tensor.device
            )

        return self._load_bytes_from_storage_with_memory(key, path, memory_obj)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return [None] * len(keys)
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error(f"Lookup failed during get_blocking for {key}")
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.hot_cache)
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)

        memory_objs: list[MemoryObj | None] = []
        xds_reads, xds_read_bytes = 0, 0
        for dtype, shape, path in zip(dtypes, shapes, paths, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt = self.fmt)
            if memory_obj is None:
                logger.error(f"Memory allocation failed during get_blocking for {path}")
            else:
                xds_reads += 1
                xds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_storage_with_memory, keys, paths, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        # logger.info(
        #     f"Time taken for batched_get_blocking: {total_time:.3f}s |"
        #     f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        # )
        return results

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_xds(
        self,
        path: str,
        kv_chunk: torch.Tensor,
        size : int,
        base_pointer: int,
        device_offset: int
    ) -> bytes:
        """ XDS 写入 """
        # 计算地址
        if base_pointer is None:
            gpu_addr = kv_chunk.data_ptr()
            dev_offset = 0
        else:
            gpu_addr = base_pointer
            dev_offset = device_offset

        tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP)

        metadata = pack_metadata(
            kv_chunk.shape,
            kv_chunk.dtype,
            size
        )

        try:
            open(tmp_path, "wb").close()
            ret = self.xds.write(
                tmp_path,
                gpu_addr,
                size,
                0,  # file_offset
                dev_offset
            )

            if ret < 0:
                raise RuntimeError(f"XDS write failed with error {ret}")
            if ret != size:
                raise RuntimeError(
                    f"XDS write incomplete: {ret}/{kv_chunk.nbytes} bytes"
                )
            # 原子性重命名
            os.rename(tmp_path, path)

        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise e

        return metadata

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_posix(self, path:str, kv_chunk:torch.Tensor, size:int, buffer):
        """ posix协议写入 """
        return self.xds.write(path, kv_chunk, size, buffer)

    def _load_xds(self, file_path: str, file_offset: int,
                  gpu_pointer: ctypes.c_void_p, size: int,
                  dev_offset: int) -> int:
        """ xds协议读取 """
        gpu_addr = gpu_pointer.value if gpu_pointer.value else 0

        ret = self.xds.read(
            file_path,
            gpu_addr,
            size,      # 实际大小
            file_offset,
            dev_offset,
        )

        if ret < 0:
            logger.error(f"XDS read failed: {file_path}, error={ret}")
        elif ret != size:
            logger.warning(f"XDS read incomplete: {file_path}, {ret}/{size} bytes")
        return ret

    def _load_posix(self, buffer, path):
        """ 读取posix """
        return self.xds.read(buffer, path)

    def touch_cache(self):
        with self.hot_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

    def pin(self, key: CacheEngineKey) -> bool:
        with self.hot_lock:
            if key in self.hot_cache:
                self.hot_cache[key].pin()
                return True
            else:
                return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.hot_lock:
            if key in self.hot_cache:
                self.hot_cache[key].unpin()
                return True
            else:
                return False

    def remove(self, key, force=True):

        with self.hot_lock:
            if not (meta := self.hot_cache.pop(key, None)):
                return False
            self.cache_policy.update_on_force_evict(key)
            # Keep the accounting consistent: external callers (StorageManager
            # batched_remove / cache_engine) reach here directly, and the old
            # code popped the entry without crediting current_cache_size back,
            # leaking `meta.size` per removed key. (_eviction no longer routes
            # through remove(), so this cannot double-subtract.)
            self.current_cache_size -= meta.size

        data_path = meta.path

        metadata_path = data_path + _METADATA_FILE_SUFFIX

        # Remove each file independently: a single try covering both meant a
        # missing/failed metadata unlink skipped the data unlink, orphaning the
        # `.kvdata` blob on the (possibly remote/shared) disk.
        for p in (metadata_path, data_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.error(f"Error removing {p} for {key}: {e}")

        # # push kv evict msg
        # if self.lmcache_worker is not None:
        #     self.lmcache_worker.put_msg(
        #         KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
        #     )

        return True

    def close(self) -> None:
        # Drain all in-flight work BEFORE tearing down the driver: cleanup()
        # first would pull the driver out from under queued puts / running
        # reads still in the worker and thread pool.
        scan_future = getattr(self, "_scan_future", None)
        if scan_future is not None:
            try:
                scan_future.result()
            except Exception as e:
                logger.error(f"XDS metadata scan failed while closing: {e}")
        self.xds_worker.close()  # drains queued put tasks (executor waits)
        self._thread_pool.shutdown(wait=True)  # drains in-flight reads
        self.xds.cleanup()
        # posix mode borrows the LocalCPU backend's allocator — only close
        # allocators we created ourselves (None on MLA non-first-rank).
        if self.config.xds_buffer_size is not None and self.memory_allocator is not None:
            self.memory_allocator.close()
        logger.info("Xds backend closed.")

    # def __del__(self):
    #     HIPFile.get_global_stats()

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata, backend: str
    ) -> HyFileMemoryAllocator:
        assert config.xds_path is not None
        assert config.xds_buffer_size is not None
        logger.info(f"use_mla:{metadata.use_mla};is_first_rank:{metadata.is_first_rank()}")
        if metadata.use_mla and not metadata.is_first_rank():
            # MLA shares one latent KV across all TP ranks (world_size==1), so
            # only the first rank offloads. Non-first ranks get no allocator and
            # the whole backend stays inert (store/get/allocate/close are guarded
            # no-ops) instead of crashing later on a None allocator / unset
            # base_pointer.
            logger.info(
                "XDS offload disabled on this rank (MLA non-first-rank): "
                "store/get/allocate are no-ops."
            )
            return None
        if backend == "hyfile":
            return HyFileMemoryAllocator(size = config.xds_buffer_size * 1024 ** 2)
        else:
            return CuFileMemoryAllocator(size = config.xds_buffer_size * 1024 ** 2)

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("Xds Backend does not support allocation with busy loop")
        # if eviction:
        #     logger.warning("Xds Backend does not support eviction")

        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return None
        return self.memory_allocator.allocate(shape, dtype, self.fmt)

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("Xds Backend does not support allocation with busy loop")
        # if eviction:
        #     logger.warning("Xds Backend does not support eviction")

        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return None
        return self.memory_allocator.batched_allocate(shape, dtype, batch_size, self.fmt)
