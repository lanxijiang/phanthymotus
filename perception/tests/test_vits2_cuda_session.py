"""CUDA session ownership and I/O regressions without loading a GPU runtime."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.runtime.backends import trt_cuda_session as backend


class FakeCuda:
    stream = 73

    def __init__(self):
        self.next_pointer = 100
        self.allocations = {}
        self.malloc_sizes = []
        self.freed = []
        self.to_device = []
        self.to_host = []
        self.sync_count = 0
        self.fail_malloc = False

    def malloc(self, size):
        if self.fail_malloc:
            raise RuntimeError("allocation failed")
        self.next_pointer += 1
        self.allocations[self.next_pointer] = np.empty(size, dtype=np.uint8)
        self.malloc_sizes.append(size)
        return self.next_pointer

    def free(self, pointer):
        if pointer:
            del self.allocations[pointer]  # Also detects a double free.
            self.freed.append(pointer)

    def copy_to_device(self, pointer, array):
        self.to_device.append(pointer)
        self.allocations[pointer][:array.nbytes] = array.reshape(-1).view(np.uint8)

    def copy_to_host(self, array, pointer):
        self.to_host.append(pointer)
        array.reshape(-1).view(np.uint8)[:] = self.allocations[pointer][:array.nbytes]

    def synchronize(self):
        self.sync_count += 1


class FakeContext:
    def __init__(self, cuda):
        self.cuda = cuda
        self.shapes = {}
        self.addresses = {}
        self.shape_calls = []
        self.fail_execution = False
        self.fail_binding = False
        self.unresolved_output = False

    def set_input_shape(self, name, shape):
        self.shape_calls.append((name, shape))
        if shape[-1] > 16:
            return False
        self.shapes[name] = shape
        return True

    def set_tensor_address(self, name, pointer):
        if self.fail_binding:
            return False
        self.addresses[name] = pointer
        return True

    def get_tensor_shape(self, name):
        if self.unresolved_output:
            return (-1,)
        return self.shapes["x"] if name == "y" else (1,)

    def execute_async_v3(self, stream):
        if self.fail_execution:
            return False
        assert stream == self.cuda.stream
        assert set(self.addresses) == {"x", "y", "sum"}
        size = int(np.prod(self.shapes["x"]))
        x = self.cuda.allocations[self.addresses["x"]][:size * 4].view(np.float32)
        for name, value in (("y", x * 2), ("sum", np.asarray([x.sum()], np.float32))):
            self.cuda.allocations[self.addresses[name]][:value.nbytes] = value.view(np.uint8)
        return True


class FakeEngine:
    num_io_tensors = 3

    def __init__(self, cuda):
        self.context = FakeContext(cuda)
        self.metadata_calls = Counter()

    def create_execution_context(self):
        return self.context

    def get_tensor_name(self, index):
        self.metadata_calls["name"] += 1
        return ("x", "y", "sum")[index]

    def get_tensor_dtype(self, name):
        self.metadata_calls["dtype"] += 1
        return np.float32

    def get_tensor_mode(self, name):
        self.metadata_calls["mode"] += 1
        return "input" if name == "x" else "output"


def make_session():
    cuda = FakeCuda()
    engine = FakeEngine(cuda)
    logger = lambda _level: None
    logger.WARNING = 2
    trt = SimpleNamespace(
        Logger=logger,
        Runtime=lambda _logger: SimpleNamespace(deserialize_cuda_engine=lambda _bytes: engine),
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )
    with patch.dict(sys.modules, {"tensorrt": trt}), \
            patch.object(Path, "read_bytes", return_value=b"fake plan"), \
            patch.object(backend, "trt_dtype_to_numpy", side_effect=lambda _trt, dtype: dtype):
        session = backend.TensorRTCudaSession("fake.plan", cuda)
    return session, cuda, engine


class TestDeviceBuffer(unittest.TestCase):
    def test_reuses_observed_capacity_and_releases_before_growing(self):
        cuda = FakeCuda()
        buffer = backend._DeviceBuffer(cuda)
        first = buffer.reserve(16)
        self.assertEqual(buffer.reserve(8), first)
        self.assertEqual(cuda.malloc_sizes, [16])
        second = buffer.reserve(32)
        self.assertNotEqual(first, second)
        self.assertEqual(cuda.freed, [first])
        self.assertEqual(sum(x.nbytes for x in cuda.allocations.values()), 32)
        buffer.close()
        buffer.close()
        self.assertFalse(cuda.allocations)

    def test_malloc_failure_clears_stale_pointer_and_can_retry(self):
        cuda = FakeCuda()
        buffer = backend._DeviceBuffer(cuda)
        first = buffer.reserve(16)
        cuda.fail_malloc = True
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            buffer.reserve(32)
        self.assertEqual((buffer.pointer, buffer.capacity), (0, 0))
        buffer.close()
        self.assertEqual(cuda.freed, [first])
        cuda.fail_malloc = False
        self.assertTrue(buffer.reserve(32))
        buffer.close()

    def test_empty_tensor_still_has_binding_and_negative_size_rejected(self):
        cuda = FakeCuda()
        buffer = backend._DeviceBuffer(cuda)
        self.assertTrue(buffer.reserve(0))
        self.assertEqual(buffer.capacity, 1)
        with self.assertRaises(ValueError):
            buffer.reserve(-1)
        buffer.close()


class TestTensorRTCudaSession(unittest.TestCase):
    def setUp(self):
        self.session, self.cuda, self.engine = make_session()
        self.addCleanup(self.session.close)

    def test_caches_metadata_and_unchanged_shapes_but_copies_new_input(self):
        for offset in (0, 10):
            x = np.arange(4, dtype=np.float32) + offset
            output = self.session.run({"x": x})
            np.testing.assert_array_equal(output["y"], x * 2)
            np.testing.assert_array_equal(output["sum"], [x.sum()])
        self.assertEqual(self.engine.metadata_calls, {"name": 3, "dtype": 3, "mode": 3})
        self.assertEqual(self.engine.context.shape_calls, [("x", (4,))])
        self.assertEqual(self.cuda.malloc_sizes, [16, 16, 4])
        self.assertEqual(self.cuda.sync_count, 2)

    def test_outputs_independently_owned_across_next_run_and_close(self):
        first = self.session.run({"x": np.arange(4)})
        before = {name: array.tobytes() for name, array in first.items()}
        second = self.session.run({"x": np.arange(8)})
        self.session.close()
        for name, array in first.items():
            self.assertTrue(array.flags.owndata)
            self.assertEqual(array.tobytes(), before[name])
            self.assertFalse(np.shares_memory(array, second[name]))
        self.assertFalse(self.cuda.allocations)

    def test_all_outputs_are_bound_and_copied_to_host(self):
        result = self.session.run({"x": np.arange(4)})
        self.assertEqual(list(result), ["y", "sum"])
        self.assertEqual(len(self.cuda.to_host), 2)
        self.assertEqual(set(self.engine.context.addresses), {"x", "y", "sum"})
        np.testing.assert_array_equal(result["y"], np.arange(4) * 2)
        np.testing.assert_array_equal(result["sum"], [6])
        self.assertEqual(self.cuda.sync_count, 1)

    def test_noncontiguous_input_casting_dynamic_resize_and_reuse(self):
        x = np.arange(8, dtype=np.float64)[::-2]
        result = self.session.run({"x": x})
        self.assertEqual(result["y"].dtype, np.float32)
        np.testing.assert_array_equal(result["y"], x.astype(np.float32) * 2)
        self.session.run({"x": np.arange(8)})
        self.session.run({"x": np.arange(2)})
        self.assertEqual(self.engine.context.shape_calls, [("x", (4,)), ("x", (8,)), ("x", (2,))])
        self.assertEqual(self.cuda.malloc_sizes, [16, 16, 4, 32, 32])

    def test_missing_input_invalid_shape_and_unresolved_output_schedule_no_copies(self):
        with self.assertRaises(KeyError):
            self.session.run({})
        with self.assertRaises(ValueError):
            self.session.run({"x": np.arange(17)})
        self.engine.context.unresolved_output = True
        with self.assertRaisesRegex(RuntimeError, "Unresolved TensorRT output"):
            self.session.run({"x": np.arange(4)})
        self.assertFalse(self.cuda.to_device)
        self.assertFalse(self.cuda.to_host)

    def test_failed_binding_never_executes_or_copies(self):
        self.engine.context.fail_binding = True
        with self.assertRaisesRegex(RuntimeError, "Failed to bind TensorRT"):
            self.session.run({"x": np.arange(4)})
        self.assertFalse(self.cuda.to_device)

    def test_execution_error_drains_stream_before_host_arrays_leave_scope(self):
        self.engine.context.fail_execution = True
        with self.assertRaisesRegex(RuntimeError, "TensorRT execution failed"):
            self.session.run({"x": np.arange(4)})
        self.assertEqual(len(self.cuda.to_device), 1)
        self.assertFalse(self.cuda.to_host)
        self.assertEqual(self.cuda.sync_count, 1)


if __name__ == "__main__":
    unittest.main()
