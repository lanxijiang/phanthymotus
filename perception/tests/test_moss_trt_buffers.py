"""Exercise buffer lifetimes and numeric output with a host-backed CUDA fixture."""

import ctypes as ct
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


class Driver:
    def __init__(self):
        self.buffers = {}
        self.events = []
        self.context = ct.c_void_p(1)
        self.stream = ct.c_void_p(2)
        self.fail_allocation = False

    def malloc(self, size):
        if self.fail_allocation:
            self.fail_allocation = False
            raise MemoryError("fixture allocation failure")
        storage = ct.create_string_buffer(size)
        address = ct.addressof(storage)
        self.buffers[address] = storage
        self.events.append(("alloc", address, size))
        return address

    def call(self, name, *args):
        self.events.append((name, *args))
        if name == "cuMemFree_v2":
            del self.buffers[args[0]]
        elif name in {"cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2"}:
            ct.memmove(*args)
        else:
            assert name in {"cuCtxSetCurrent", "cuStreamSynchronize"}


class Engine:
    names = ("signal", "scale", "audio", "length")
    num_io_tensors = len(names)

    def get_tensor_name(self, index):
        return self.names[index]

    def get_tensor_mode(self, name):
        return "input" if name in {"signal", "scale"} else "output"

    def get_tensor_dtype(self, name):
        return np.int32 if name == "length" else np.float32

    def get_tensor_location(self, name):
        return "host" if name in {"scale", "length"} else "device"


class Context:
    def __init__(self):
        self.addresses = {}
        self.shapes = {}
        self.fail_execute = False

    def set_input_shape(self, name, shape):
        self.shapes[name] = shape
        return shape == () if name == "scale" else len(shape) == 1 and shape[0] <= 8

    def set_tensor_address(self, name, address):
        self.addresses[name] = address
        return True

    def infer_shapes(self):
        return []

    def get_tensor_shape(self, name):
        return () if name == "length" else self.shapes["signal"]

    def execute_async_v3(self, stream):
        if self.fail_execute:
            return False
        length = self.shapes["signal"][0]
        array_type = ct.c_float * length
        signal = np.ctypeslib.as_array(array_type.from_address(self.addresses["signal"]))
        audio = np.ctypeslib.as_array(array_type.from_address(self.addresses["audio"]))
        audio[:] = signal * ct.c_float.from_address(self.addresses["scale"]).value
        ct.c_int32.from_address(self.addresses["length"]).value = length
        return True


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(
        TensorIOMode=SimpleNamespace(INPUT="input", OUTPUT="output"),
        TensorLocation=SimpleNamespace(HOST="host"), nptype=lambda dtype: dtype,
    ))
    path = Path(__file__).resolve().parents[1] / "tools/moss_trt_session.py"
    spec = importlib.util.spec_from_file_location("_trt_buffer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session(runner, monkeypatch, *, enabled=True, graph="local_cached_step", cuda=None):
    if enabled:
        monkeypatch.setenv("MOSS_TRT_REUSE_LOCAL_BUFFERS", "1")
    else:
        monkeypatch.delenv("MOSS_TRT_REUSE_LOCAL_BUFFERS", raising=False)
    result = object.__new__(runner.TensorRTSession)
    result.cuda = cuda or Driver()
    result.engine = Engine()
    result.context = Context()
    result.configure_profiling(graph, enabled=True)
    return result


def run(model, values):
    result = model.run({"signal": np.asarray(values, dtype=np.float32), "scale": np.array(2, dtype=np.float32)})
    np.testing.assert_array_equal(result["audio"], np.asarray(values) * 2)
    assert result["length"].shape == ()
    assert result["length"].item() == len(values)
    return result


def test_reuse_plateaus_preserves_old_outputs_and_closes(runner, monkeypatch):
    model = session(runner, monkeypatch)
    first = run(model, [1, 2])["audio"]
    run(model, [4, 5])
    np.testing.assert_array_equal(first, [2, 4])
    assert model.profile_counters["cuda_allocations"] == 2
    assert model.profile_counters["cuda_frees"] == 0
    model.configure_profiling("local_cached_step", enabled=True)
    run(model, [5, 6])
    assert model.profile_counters["cuda_allocations"] == 0
    model.close()
    assert not model.cuda.buffers
    assert model.profile_counters["cuda_frees"] == 2
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        run(model, [1])


@pytest.mark.parametrize("enabled,graph", [(False, "local_cached_step"), (True, "global_unified"), (True, "codec_decode_step")])
def test_default_and_other_graphs_free_every_run(runner, monkeypatch, enabled, graph):
    model = session(runner, monkeypatch, enabled=enabled, graph=graph)
    run(model, [1, 2])
    run(model, [3, 4])
    assert model.profile_counters["cuda_allocations"] == 4
    assert model.profile_counters["cuda_frees"] == 4
    assert not model.cuda.buffers
    assert not model._device_buffers


def test_resize_zero_grow_shrink_and_largest_shape(runner, monkeypatch):
    model = session(runner, monkeypatch)
    run(model, [])
    run(model, [1, 2])
    assert model.profile_counters["cuda_frees"] == 2
    previous = run(model, list(range(8)))["audio"]
    allocation_count = model.profile_counters["cuda_allocations"]
    run(model, [9])
    run(model, [])
    assert model.profile_counters["cuda_allocations"] == allocation_count
    np.testing.assert_array_equal(previous, np.arange(8) * 2)
    model.close()
    assert not model.cuda.buffers


@pytest.mark.parametrize("failure", ["execute", "shape", "allocate"])
def test_failed_run_releases_all_cached_buffers_then_can_retry(runner, monkeypatch, failure):
    model = session(runner, monkeypatch)
    run(model, [1, 2])
    model.cuda.events.clear()
    if failure == "execute":
        model.context.fail_execute = True
        values, error = [3, 4], RuntimeError
    elif failure == "shape":
        values, error = list(range(9)), ValueError
    else:
        model.cuda.fail_allocation = True
        values, error = [3, 4, 5], MemoryError
    with pytest.raises(error):
        run(model, values)
    assert not model.cuda.buffers
    assert not model._device_buffers
    assert model.profile_counters["errors"] == 1
    if failure == "execute":
        names = [event[0] for event in model.cuda.events]
        assert names.index("cuStreamSynchronize") < names.index("cuMemFree_v2")
    model.context.fail_execute = False
    run(model, [7])
    model.close()


def test_sessions_never_share_writable_device_buffers(runner, monkeypatch):
    cuda = Driver()
    models = [session(runner, monkeypatch, cuda=cuda) for _ in range(3)]
    results = [run(model, [index, index + 1])["audio"] for index, model in enumerate(models)]
    assert len(cuda.buffers) == 6
    for index, model in enumerate(models):
        run(model, [20 + index, 30 + index])
        np.testing.assert_array_equal(results[index], np.array([index, index + 1]) * 2)
    for model in models:
        model.close()
    assert not cuda.buffers


def test_pool_shutdown_releases_buffers_before_destroying_stream(runner, monkeypatch):
    from plugins.moss_deploy.trt_pool import TensorRTGraphPool

    model = session(runner, monkeypatch)
    run(model, [1, 2])
    cuda = model.cuda
    loader_cuda = Driver()
    for driver in (cuda, loader_cuda):
        driver.lib = SimpleNamespace()
        driver.device = ct.c_int(0)
        driver._bind = lambda *args: None
        original_call = driver.call

        def call(name, *args, driver=driver, original_call=original_call):
            if name in {"cuStreamDestroy_v2", "cuDevicePrimaryCtxRelease"}:
                driver.events.append((name, *args))
            else:
                original_call(name, *args)

        driver.call = call
    pool = object.__new__(TensorRTGraphPool)
    pool.closed = False
    pool.loader_cuda = loader_cuda
    pool.shared_encoder_session = None
    pool.shared_encoder_cuda = None
    pool.context_inventory_lock = threading.Lock()
    pool.instance_sessions = {"one": {"local_cached_step": model}}
    pool.instance_cudas = {"one": cuda}
    pool.engines = {}
    pool.trt_runtime = None
    pool.logger = None
    pool.close()
    assert not cuda.buffers
    assert model._closed
    names = [event[0] for event in cuda.events]
    assert names.index("cuMemFree_v2") < names.index("cuStreamDestroy_v2")
    pool.close()


class FrameEngine:
    shapes = {"global_hidden": (1, 768), "random_u": (1, 17),
              "should_continue": (1, 1), "frame_token_ids": (1, 16),
              "text_logits": (1, 2), "audio_logits": (1, 16, 1024)}
    names = tuple(shapes)
    num_io_tensors = len(names)

    def get_tensor_name(self, index):
        return self.names[index]

    def get_tensor_mode(self, name):
        return "input" if name in {"global_hidden", "random_u"} else "output"

    def get_tensor_dtype(self, name):
        return np.int32 if name in {"should_continue", "frame_token_ids"} else np.float32

    def get_tensor_location(self, name):
        return "device"

    def get_tensor_shape(self, name):
        return self.shapes[name]


class FrameContext(Context):
    def set_input_shape(self, name, shape):
        self.shapes[name] = shape
        return shape == FrameEngine.shapes[name]

    def get_tensor_shape(self, name):
        return FrameEngine.shapes[name]

    def execute_async_v3(self, stream):
        if self.fail_execute:
            return False
        hidden = ct.c_float.from_address(self.addresses["global_hidden"]).value
        random = ct.c_float.from_address(self.addresses["random_u"]).value
        for name in FrameEngine.names[2:]:
            dtype = ct.c_int32 if FrameEngine().get_tensor_dtype(name) == np.int32 else ct.c_float
            length = int(np.prod(FrameEngine.shapes[name]))
            values = np.ctypeslib.as_array((dtype * length).from_address(self.addresses[name]))
            values[:] = int(hidden) % 2 if name == "should_continue" else int(hidden * 100 + random * 10)
        return True


def frame_session(runner, *, cuda=None):
    model = object.__new__(runner.TensorRTSession)
    model.cuda, model.engine, model.context = cuda or Driver(), FrameEngine(), FrameContext()
    model.configure_profiling("local_fixed_sampled_frame", enabled=True)
    return model


def run_frame(model, value):
    result = model.run({"global_hidden": np.full((1, 768), value, dtype=np.float32),
                        "random_u": np.full((1, 17), .25, dtype=np.float32)})
    for name, array in result.items():
        assert array.shape == FrameEngine.shapes[name]
        assert array.dtype == FrameEngine().get_tensor_dtype(name)
        np.testing.assert_array_equal(array, int(value) % 2 if name == "should_continue" else int(value * 100 + 2.5))
    return result


@pytest.mark.parametrize("environment,enabled", [(None, True), ("0", False), ("1", True)])
def test_fused_fixed_io_buffers_reuse_by_default_with_explicit_opt_out(runner, monkeypatch, environment, enabled):
    monkeypatch.delenv("MOSS_TRT_REUSE_LOCAL_BUFFERS", raising=False)
    monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", "1")
    if environment is None:
        monkeypatch.delenv("MOSS_TRT_REUSE_FRAME_BUFFERS", raising=False)
    else:
        monkeypatch.setenv("MOSS_TRT_REUSE_FRAME_BUFFERS", environment)
    model = frame_session(runner)
    assert model._reuse_device_buffers is enabled
    assert model._resident_enabled is False  # No external state pairs exist.
    first = run_frame(model, 1)
    preserved = {name: value.copy() for name, value in first.items()}
    first_addresses = dict(model.context.addresses)
    # Resetting counters or changing env after construction cannot change the
    # buffer ownership policy underneath an already initialized session.
    monkeypatch.setenv("MOSS_TRT_REUSE_FRAME_BUFFERS", "0" if enabled else "1")
    model.configure_profiling("local_fixed_sampled_frame", enabled=True)
    second = run_frame(model, 2)
    for name in first:
        np.testing.assert_array_equal(first[name], preserved[name])
        assert not np.shares_memory(first[name], second[name])
    assert model.profile_counters["cuda_allocations"] == (0 if enabled else 6)
    assert model.profile_counters["cuda_frees"] == (0 if enabled else 6)
    if enabled:
        assert model.context.addresses == first_addresses
    model.close()
    assert not model.cuda.buffers


@pytest.mark.parametrize("failure", ["execute", "shape"])
def test_fused_failed_execution_reclaims_all_six_buffers_before_retry(runner, monkeypatch, failure):
    monkeypatch.delenv("MOSS_TRT_REUSE_FRAME_BUFFERS", raising=False)
    model = frame_session(runner)
    run_frame(model, 1)
    assert len(model.cuda.buffers) == 6
    model.cuda.events.clear()
    if failure == "execute":
        model.context.fail_execute = True
        with pytest.raises(RuntimeError, match="execution failed"):
            run_frame(model, 2)
    else:
        with pytest.raises(ValueError, match="Unsupported shape"):
            model.run({"global_hidden": np.zeros((2, 768), dtype=np.float32),
                       "random_u": np.zeros((1, 17), dtype=np.float32)})
    assert not model.cuda.buffers and not model._device_buffers
    names = [event[0] for event in model.cuda.events]
    assert names.index("cuStreamSynchronize") < names.index("cuMemFree_v2")
    model.context.fail_execute = False
    run_frame(model, 3)
    assert len(model.cuda.buffers) == 6
    model.close()
    assert not model.cuda.buffers


def test_fused_three_contexts_keep_separate_reused_io_buffers(runner, monkeypatch):
    monkeypatch.delenv("MOSS_TRT_REUSE_FRAME_BUFFERS", raising=False)
    cuda = Driver()
    models = [frame_session(runner, cuda=cuda) for _ in range(3)]
    first_results = [run_frame(model, index) for index, model in enumerate(models)]
    assert len(cuda.buffers) == 18
    pointer_sets = [set(model.context.addresses.values()) for model in models]
    assert all(not pointer_sets[i] & pointer_sets[j] for i in range(3) for j in range(i))
    for index, model in enumerate(models):
        run_frame(model, index + 5)
        np.testing.assert_array_equal(first_results[index]["audio_logits"], int(index * 100 + 2.5))
    for model in models:
        model.close()
    assert not cuda.buffers


class GraphDriver(Driver):
    """Queue async copies/work and capture callbacks without running them."""

    def __init__(self):
        super().__init__()
        self.host_buffers, self.streams, self.graphs, self.executables = {}, {}, {}, {}
        self.capturing, self.captured = None, None
        self.next_handle, self.fail_once = 100, None

    def bind_graph_api(self):
        self.events.append(("bind_graph_api",))

    def handle(self):
        self.next_handle += 1
        return self.next_handle

    def call(self, name, *args):
        self.events.append((name, *(x.value if isinstance(x, ct.c_void_p) else x for x in args)))
        if self.fail_once == name:
            self.fail_once = None
            if name == "cuStreamEndCapture":
                self.capturing, self.captured = None, None
            raise RuntimeError(f"fixture {name} failure")
        if name == "cuStreamCreate":
            assert args[1] == 1, "Graph streams must be nonblocking"
            handle = self.handle()
            args[0]._obj.value = handle
            self.streams[handle] = []
        elif name == "cuMemAllocHost_v2":
            buffer = ct.create_string_buffer(args[1])
            address = ct.addressof(buffer)
            self.host_buffers[address] = buffer
            args[0]._obj.value = address
        elif name == "cuMemFreeHost":
            del self.host_buffers[args[0].value]
        elif name == "cuMemcpyHtoDAsync_v2":
            assert self.capturing is None, "Capture must contain only TRT work"
            assert args[1].value in self.host_buffers
            self.streams[args[3].value].append(lambda: ct.memmove(*args[:3]))
        elif name == "cuStreamBeginCapture_v2":
            assert not self.streams[args[0].value], "Warmup must finish before capture"
            assert args[1] == 1  # THREAD_LOCAL
            assert self.capturing is None
            self.capturing, self.captured = args[0].value, None
        elif name == "cuStreamEndCapture":
            assert self.capturing == args[0].value
            handle = self.handle()
            self.graphs[handle] = self.captured
            args[1]._obj.value = handle
            self.capturing, self.captured = None, None
        elif name == "cuGraphInstantiateWithFlags":
            handle = self.handle()
            self.executables[handle] = self.graphs[args[1].value]
            args[0]._obj.value = handle
        elif name == "cuGraphLaunch":
            assert self.capturing is None
            self.streams[args[1].value].append(self.executables[args[0].value])
        elif name == "cuStreamSynchronize":
            work = self.streams.get(args[0].value, [])
            while work:
                work.pop(0)()
        elif name == "cuGraphExecDestroy":
            del self.executables[args[0].value]
        elif name == "cuGraphDestroy":
            del self.graphs[args[0].value]
        elif name == "cuStreamDestroy_v2":
            assert not self.streams.pop(args[0].value)
        elif name == "cuMemcpyDtoH_v2":
            assert self.capturing is None
            assert not any(self.streams.values()), "Output must wait for graph completion"
            ct.memmove(*args)
        elif name == "cuMemFree_v2":
            assert not any(self.streams.values()), "Cannot free memory with queued work"
            del self.buffers[args[0]]
        else:
            assert name == "cuCtxSetCurrent"


class GraphFrameContext(FrameContext):
    def __init__(self, cuda):
        super().__init__()
        self.cuda = cuda
        self.enqueues, self.shape_updates, self.address_updates, self.aux_updates = [], [], [], []
        self.fail_capture = False

    def set_input_shape(self, name, shape):
        self.shape_updates.append((name, shape))
        return super().set_input_shape(name, shape)

    def set_tensor_address(self, name, address):
        self.address_updates.append((name, address))
        return super().set_tensor_address(name, address)

    def set_aux_streams(self, streams):
        assert len(set(streams)) == len(streams)
        assert all(stream in self.cuda.streams for stream in streams)
        self.aux_updates.append(tuple(streams))

    def execute_async_v3(self, stream):
        self.enqueues.append((stream, self.cuda.capturing is not None))
        if self.fail_execute or (self.fail_capture and self.cuda.capturing is not None):
            return False
        work = lambda: FrameContext.execute_async_v3(self, stream)
        if self.cuda.capturing is not None:
            assert self.cuda.capturing == stream
            self.cuda.captured = work
        else:
            self.cuda.streams[stream].append(work)
        return True


def graph_session(runner, monkeypatch, *, cuda=None, profile=True, aux=0):
    monkeypatch.setenv("MOSS_TRT_FRAME_CUDA_GRAPH", "1")
    model = frame_session(runner, cuda=cuda or GraphDriver())
    model.context = GraphFrameContext(model.cuda)
    model.engine.num_aux_streams = aux
    model.configure_profiling("local_fixed_sampled_frame", enabled=profile)
    return model


def assert_graph_released(model):
    cuda = model.cuda
    assert not (cuda.buffers or cuda.host_buffers or cuda.streams or cuda.graphs or cuda.executables)
    assert cuda.capturing is None
    assert not model.cuda_graph_snapshot()["ready"]


def test_cuda_graph_driver_binds_versioned_abi_once(runner):
    driver = object.__new__(runner.CudaDriver)
    bound = []
    driver._bind = lambda name, args: bound.append((name, len(args)))
    driver.bind_graph_api()
    expected = list(bound)
    driver.bind_graph_api()
    assert bound == expected
    assert ("cuStreamBeginCapture_v2", 2) in bound
    assert ("cuGraphInstantiateWithFlags", 3) in bound
    assert ("cuMemAllocHost_v2", 2) in bound
    assert ("cuMemcpyHtoDAsync_v2", 4) in bound
    assert not {"cuStreamBeginCapture", "cuGraphInstantiate"} & {name for name, _ in bound}


def test_cuda_graph_is_opt_in_and_never_applies_to_stable_graph(runner, monkeypatch):
    monkeypatch.delenv("MOSS_TRT_FRAME_CUDA_GRAPH", raising=False)
    fused = frame_session(runner)
    run_frame(fused, 1)
    assert not fused.cuda_graph_snapshot()["enabled"]
    fused.close()
    monkeypatch.setenv("MOSS_TRT_FRAME_CUDA_GRAPH", "1")
    stable = session(runner, monkeypatch)
    run(stable, [1, 2])
    assert not stable.cuda_graph_snapshot()["enabled"]
    stable.close()
    monkeypatch.setenv("MOSS_TRT_REUSE_FRAME_BUFFERS", "0")
    with pytest.raises(ValueError, match="REUSE_FRAME_BUFFERS=1"):
        frame_session(runner)


@pytest.mark.parametrize("profile", [False, True])
def test_cuda_graph_replays_new_inputs_without_enqueue_or_rebinding(runner, monkeypatch, profile):
    model = graph_session(runner, monkeypatch, profile=profile, aux=2)
    first = run_frame(model, 1)
    status = model.cuda_graph_snapshot()
    assert status == {"enabled": True, "ready": True, "closed": False, "captures": 1,
                      "replays": 0, "warmup_enqueues": 1, "capture_failures": 0, "last_error": None}
    assert len(model.context.enqueues) == 2  # Warmup then capture; capture does not run.
    assert model.context.enqueues[0][1] is False and model.context.enqueues[1][1] is True
    assert len(model.context.aux_updates) == 2  # set_aux_streams only affects the next enqueue.
    assert model.context.aux_updates[0] == model.context.aux_updates[1]
    assert len(model.cuda.buffers) == 6 and len(model.cuda.host_buffers) == 2
    assert model.cuda.stream.value == 2  # The legacy stream is unchanged.
    model.configure_profiling("local_fixed_sampled_frame", enabled=profile)
    model.cuda.events.clear()
    for value in [2, 3]:
        latest = run_frame(model, value)
        assert all(not np.shares_memory(first[name], latest[name]) for name in first)
    np.testing.assert_array_equal(first["audio_logits"], 102)
    assert len(model.context.enqueues) == 2
    assert len(model.context.shape_updates) == 2 and len(model.context.address_updates) == 6
    assert model.cuda_graph_snapshot()["captures"] == 1
    assert model.cuda_graph_snapshot()["replays"] == 2
    assert not any(event[0] in {"alloc", "cuMemAllocHost_v2"} for event in model.cuda.events)
    if profile:
        counters = model.profile_snapshot()
        assert counters["cuda_graph_captures"] == 0  # Timings reset, lifetime snapshot does not.
        assert counters["cuda_graph_replays"] == 2
        assert counters["cuda_allocations"] == 0
        assert counters["cuda_graph_ready"] == 1
    else:
        assert model.profile_snapshot() is None
    model.cuda.events.clear()
    model.close()
    events = [event[0] for event in model.cuda.events]
    assert events.index("cuStreamSynchronize") < events.index("cuGraphExecDestroy")
    assert events.index("cuGraphExecDestroy") < events.index("cuMemFree_v2")
    assert events.index("cuGraphDestroy") < events.index("cuMemFreeHost")
    assert_graph_released(model)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        run_frame(model, 1)


@pytest.mark.parametrize("failure", ["warmup", "capture_enqueue", "cuStreamBeginCapture_v2",
                                     "cuStreamEndCapture", "cuGraphInstantiateWithFlags",
                                     "cuMemcpyHtoDAsync_v2", "cuMemAllocHost_v2", "cuGraphLaunch"])
def test_cuda_graph_failure_is_explicit_releases_work_and_allows_fresh_capture(runner, monkeypatch, failure):
    model = graph_session(runner, monkeypatch)
    if failure == "cuGraphLaunch":
        run_frame(model, 1)
    if failure == "warmup":
        model.context.fail_execute = True
    elif failure == "capture_enqueue":
        model.context.fail_capture = True
    else:
        model.cuda.fail_once = failure
    with pytest.raises(RuntimeError, match="(fixture|enqueue failed)"):
        run_frame(model, 2)
    assert_graph_released(model)
    assert model.cuda_graph_snapshot()["last_error"]
    assert model.profile_counters["errors"] == 1
    model.context.fail_execute = model.context.fail_capture = False
    run_frame(model, 3)
    run_frame(model, 4)
    assert model.cuda_graph_snapshot()["ready"]
    model.close()
    assert_graph_released(model)


def test_cuda_graph_wrong_shape_drops_capture_and_rejects_dynamic_engine(runner, monkeypatch):
    model = graph_session(runner, monkeypatch)
    run_frame(model, 1)
    with pytest.raises(ValueError, match="dimensions cannot change"):
        model.run({"global_hidden": np.zeros((2, 768)), "random_u": np.zeros((1, 17))})
    assert_graph_released(model)
    model.engine.get_tensor_shape = lambda name: (-1, 768) if name == "global_hidden" else FrameEngine.shapes[name]
    with pytest.raises(ValueError, match="Unsupported fixed CUDA Graph tensor: global_hidden"):
        run_frame(model, 1)
    assert_graph_released(model)
    model.close()


def test_cuda_graph_three_sessions_own_streams_graphs_staging_and_buffers(runner, monkeypatch):
    cuda = GraphDriver()
    models = [graph_session(runner, monkeypatch, cuda=cuda) for _ in range(3)]
    previous = [run_frame(model, index) for index, model in enumerate(models)]
    assert len({model._graph_stream.value for model in models}) == 3
    assert len({model._graph_exec.value for model in models}) == 3
    assert len(cuda.buffers) == 18 and len(cuda.host_buffers) == 6
    for index, model in enumerate(models):
        run_frame(model, index + 10)
        np.testing.assert_array_equal(previous[index]["audio_logits"], index * 100 + 2)
    for model in models:
        model.close()
    assert_graph_released(models[0])


def test_pool_memory_reports_actual_cuda_graph_state_when_profile_is_off(runner, monkeypatch):
    from plugins.moss_deploy.trt_pool import TensorRTGraphPool

    model = graph_session(runner, monkeypatch, profile=False)
    run_frame(model, 1)
    run_frame(model, 2)
    pool = object.__new__(TensorRTGraphPool)
    pool.closed, pool.profile_enabled = False, False
    pool.loader_cuda = model.cuda
    pool.loader_cuda.info = lambda: {"free_bytes": 1234}
    pool.context_inventory_lock = threading.Lock()
    pool.instance_sessions = {"test": {"local_fixed_sampled_frame": model,
                                       "legacy_mock": SimpleNamespace()}}
    pool.shared_encoder_session = None
    pool.encoder_context_mode = "disabled"
    pool.omitted_graphs = {"codec_encoder": "disabled"}
    pool.engines = {"local_fixed_sampled_frame": model.engine}
    assert pool.performance_snapshot() is None
    report = pool.memory_snapshot()
    assert report["cuda_graph_sessions"] == [dict(model.cuda_graph_snapshot(), instance_id="test",
                                                 graph="local_fixed_sampled_frame")]
    assert report["cuda_graph_sessions"][0]["captures"] == 1
    assert report["cuda_graph_sessions"][0]["replays"] == 1
    model.close()
