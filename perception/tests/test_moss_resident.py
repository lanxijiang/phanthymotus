"""Validate resident state ownership, chaining, resets and cleanup offline."""

import ctypes as ct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_moss_trt_buffers import Driver, runner


class StateEngine:
    def __init__(self, graph):
        if graph == "global_unified":
            self.key, self.value = "past_key_0", "past_value_0"
            self.out_key, self.out_value = "present_key_0", "present_value_0"
        elif graph == "local_cached_step":
            self.key, self.value = "local_past_key_0", "local_past_value_0"
            self.out_key, self.out_value = "local_present_key_0", "local_present_value_0"
        else:
            self.key, self.value = "attention_0_keys", "attention_0_values"
            self.out_key, self.out_value = "present_attention_0_keys", "present_attention_0_values"
        self.inputs = (self.key, self.value, "token", "scale")
        self.names = (*self.inputs, self.out_key, self.out_value, "audio", "length")
        self.num_io_tensors = len(self.names)

    def get_tensor_name(self, index):
        return self.names[index]

    def get_tensor_mode(self, name):
        return "input" if name in self.inputs else "output"

    def get_tensor_dtype(self, name):
        return np.int32 if name == "length" else np.float32

    def get_tensor_location(self, name):
        return "host" if name in {"scale", "length"} else "device"


class StateContext:
    def __init__(self, engine):
        self.engine = engine
        self.addresses, self.shapes = {}, {}
        self.executions = []
        self.fail_execute = self.fail_infer = False

    def set_input_shape(self, name, shape):
        self.shapes[name] = shape
        if name in {self.engine.key, self.engine.value}:
            return len(shape) == 1 and shape[0] <= 8
        return shape == ()

    def set_tensor_address(self, name, address):
        self.addresses[name] = address
        return True

    def infer_shapes(self):
        return ["fixture_unresolved"] if self.fail_infer else []

    def get_tensor_shape(self, name):
        if name == self.engine.out_key:
            return (self.shapes[self.engine.key][0] + 1,)
        if name == self.engine.out_value:
            return (self.shapes[self.engine.value][0] + 1,)
        return () if name == "length" else (1,)

    def array(self, name, length):
        return np.ctypeslib.as_array((ct.c_float * length).from_address(self.addresses[name]))

    def execute_async_v3(self, stream):
        if self.fail_execute:
            return False
        self.executions.append(self.addresses.copy())
        engine = self.engine
        token = ct.c_float.from_address(self.addresses["token"]).value
        scale = ct.c_float.from_address(self.addresses["scale"]).value
        total = 0
        for source, output, item in ((engine.key, engine.out_key, token), (engine.value, engine.out_value, token * scale)):
            assert self.addresses[source] != self.addresses[output], "State input/output cannot alias"
            length = self.shapes[source][0]
            destination = self.array(output, length + 1)
            destination[:-1] = self.array(source, length)
            destination[-1] = item
            total += destination.sum()
        self.array("audio", 1)[0] = total
        ct.c_int32.from_address(self.addresses["length"]).value = self.shapes[engine.key][0] + 1
        return True


def make_session(runner, monkeypatch, graph="local_cached_step", enabled=True):
    monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", "1" if enabled else "0")
    monkeypatch.delenv("MOSS_TRT_REUSE_LOCAL_BUFFERS", raising=False)
    session = object.__new__(runner.TensorRTSession)
    session.cuda = Driver()
    session.engine = StateEngine(graph)
    session.context = StateContext(session.engine)
    session.configure_profiling(graph, enabled=True)
    return session


def feeds(session, state=None, token=1):
    engine = session.engine
    key, value = state if state is not None else (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    return {engine.key: key, engine.value: value, "token": np.array(token, dtype=np.float32), "scale": np.array(2, dtype=np.float32)}


def state(session, result):
    return result[session.engine.out_key], result[session.engine.out_value]


@pytest.mark.parametrize("graph", ["global_unified", "local_cached_step", "codec_decode_step"])
def test_two_bank_chaining_matches_numpy_path_without_state_copies(runner, monkeypatch, graph):
    resident = make_session(runner, monkeypatch, graph)
    ordinary = make_session(runner, monkeypatch, graph, enabled=False)
    gpu_state = cpu_state = None
    retained_audio = []
    for token in range(1, 6):
        resident.cuda.events.clear()
        actual = resident.run(feeds(resident, gpu_state, token))
        expected = ordinary.run(feeds(ordinary, cpu_state, token))
        gpu_state, cpu_state = state(resident, actual), state(ordinary, expected)
        # Before explicit validation materialization, only token/audio is copied.
        uploads = [event for event in resident.cuda.events if event[0] == "cuMemcpyHtoD_v2"]
        downloads = [event for event in resident.cuda.events if event[0] == "cuMemcpyDtoH_v2"]
        assert sum(event[-1] for event in uploads) == 4
        assert sum(event[-1] for event in downloads) == 4
        for actual_state, expected_state in zip(gpu_state, cpu_state):
            assert isinstance(actual_state, runner.DeviceTensor)
            np.testing.assert_array_equal(actual_state.materialize(), expected_state)
        np.testing.assert_array_equal(actual["audio"], expected["audio"])
        assert actual["length"].shape == ()
        assert actual["length"].item() == token
        retained_audio.append(actual["audio"])
        for index, audio in enumerate(retained_audio, 1):
            assert audio[0] == 3 * index * (index + 1) / 2
    resident.close()
    ordinary.close()
    assert not resident.cuda.buffers
    assert not ordinary.cuda.buffers


def test_explicit_host_reset_uploads_new_state_and_invalidates_old_handles(runner, monkeypatch):
    session = make_session(runner, monkeypatch)
    previous = session.run(feeds(session, token=3))
    old_state = state(session, previous)
    session.cuda.events.clear()
    reset = session.run(feeds(session, (np.array([10., 20.]), np.array([30., 40.])), 2))
    np.testing.assert_array_equal(state(session, reset)[0].materialize(), [10, 20, 2])
    np.testing.assert_array_equal(state(session, reset)[1].materialize(), [30, 40, 4])
    uploads = [event for event in session.cuda.events if event[0] == "cuMemcpyHtoD_v2"]
    assert sum(event[-1] for event in uploads) == 20
    assert previous["audio"][0] == 9
    for handle in old_state:
        with pytest.raises(ValueError, match="Stale"):
            handle.materialize()
    empty_reset = session.run(feeds(session, token=8))
    np.testing.assert_array_equal(state(session, empty_reset)[0].materialize(), [8])
    session.close()


@pytest.mark.parametrize("invalid", ["stale", "foreign", "wrong_pair", "names"])
def test_invalid_borrow_does_not_invalidate_existing_good_state(runner, monkeypatch, invalid):
    session = make_session(runner, monkeypatch)
    other = make_session(runner, monkeypatch)
    first = session.run(feeds(session, token=1))
    stale = state(session, first)
    second = session.run(feeds(session, stale, 2))
    third = session.run(feeds(session, state(session, second), 3))
    good = state(session, third)
    expected = [handle.materialize().copy() for handle in good]
    foreign = state(other, other.run(feeds(other, token=9)))
    supplied = feeds(session, good, 4)
    if invalid == "stale":
        supplied[session.engine.key] = stale[0]
    elif invalid == "foreign":
        supplied[session.engine.key] = foreign[0]
    elif invalid == "wrong_pair":
        supplied[session.engine.key] = good[1]
    else:
        supplied["unexpected"] = np.array(0)
    before = set(session.cuda.buffers)
    with pytest.raises(ValueError):
        session.run(supplied)
    assert set(session.cuda.buffers) == before
    for handle, snapshot in zip(good, expected):
        np.testing.assert_array_equal(handle.materialize(), snapshot)
    np.testing.assert_array_equal(foreign[0].materialize(), [9])
    session.close()
    other.close()


@pytest.mark.parametrize("failure", ["execute", "shape", "allocate", "infer"])
def test_failed_execution_frees_all_buffers_and_fresh_state_can_retry(runner, monkeypatch, failure):
    session = make_session(runner, monkeypatch)
    original = session.run(feeds(session, token=1))
    handles = state(session, original)
    supplied = feeds(session, handles, 2)
    expected_error = RuntimeError
    if failure == "execute":
        session.context.fail_execute = True
    elif failure == "shape":
        supplied[session.engine.key] = np.zeros(9, dtype=np.float32)
        expected_error = ValueError
    elif failure == "allocate":
        session.cuda.fail_allocation = True
        expected_error = MemoryError
    else:
        session.context.fail_infer = True
    with pytest.raises(expected_error):
        session.run(supplied)
    assert not session.cuda.buffers
    assert not session._state_slots
    assert not session._device_buffers
    assert not session._bound_addresses
    assert not session._bound_shapes
    for handle in handles:
        with pytest.raises(ValueError, match="Stale"):
            handle.materialize()
    session.context.fail_execute = session.context.fail_infer = False
    recovered = session.run(feeds(session, token=7))
    np.testing.assert_array_equal(state(session, recovered)[0].materialize(), [7])
    session.close()


def test_close_invalidates_state_and_implicit_copy_is_forbidden(runner, monkeypatch):
    session = make_session(runner, monkeypatch)
    handle = state(session, session.run(feeds(session)))[0]
    with pytest.raises(TypeError, match="implicitly copied"):
        np.asarray(handle)
    session.close()
    assert not session.cuda.buffers
    with pytest.raises(ValueError, match="closed"):
        handle.materialize()
    session.close()


@pytest.mark.parametrize("environment,expected", [(None, False), ("0", False), ("1", True)])
def test_generic_constructor_keeps_resident_opt_in(runner, monkeypatch, tmp_path, environment, expected):
    if environment is None:
        monkeypatch.delenv("MOSS_TRT_RESIDENT_STATE", raising=False)
    else:
        monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", environment)
    engine = StateEngine("local_cached_step")
    engine.create_execution_context = lambda: StateContext(engine)
    logger = type("Logger", (), {"WARNING": 1, "__init__": lambda self, level: None})
    monkeypatch.setattr(runner.trt, "Logger", logger, raising=False)
    monkeypatch.setattr(runner.trt, "Runtime", lambda log: SimpleNamespace(
        deserialize_cuda_engine=lambda data: engine), raising=False)
    plan = tmp_path / "local_cached_step.plan"
    plan.write_bytes(b"fixture")
    model = runner.TensorRTSession(plan, Driver())
    result = model.run(feeds(model))
    assert model._resident_enabled is expected
    for output in state(model, result):
        assert isinstance(output, runner.DeviceTensor if expected else np.ndarray)
    model.close()


@pytest.mark.parametrize("environment,expected", [(None, True), ("0", False), ("1", True)])
def test_pool_freezes_default_for_all_lazy_instance_sessions(runner, monkeypatch, tmp_path, environment, expected):
    from plugins.moss_deploy import trt_pool

    if environment is None:
        monkeypatch.delenv("MOSS_TRT_RESIDENT_STATE", raising=False)
    else:
        monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", environment)

    class PoolDriver(Driver):
        def __init__(self, device):
            super().__init__()
            self.device, self.lib = device, SimpleNamespace()

        def _bind(self, *args):
            pass

        def call(self, name, *args):
            if name in {"cuStreamDestroy_v2", "cuDevicePrimaryCtxRelease"}:
                self.events.append((name, *args))
            else:
                super().call(name, *args)

    engine = StateEngine("global_unified")
    engine.create_execution_context = lambda: StateContext(engine)
    trt = sys.modules["tensorrt"]
    monkeypatch.setattr(trt, "Logger", type("Logger", (), {
        "WARNING": 1, "__init__": lambda self, level: None}), raising=False)
    monkeypatch.setattr(trt, "Runtime", lambda log: SimpleNamespace(
        deserialize_cuda_engine=lambda data: engine), raising=False)
    monkeypatch.setattr(runner, "CudaDriver", PoolDriver)
    monkeypatch.setattr(trt_pool.importlib.util, "spec_from_file_location", lambda *args:
                        SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(trt_pool.importlib.util, "module_from_spec", lambda spec: runner)
    plan = tmp_path / "global_unified.plan"
    plan.write_bytes(b"fixture")
    assets = SimpleNamespace(graph_paths={"global_unified": plan}, streaming_path=None,
                             precomputed_reference_codes={}, voice_paths={})
    pool = trt_pool.TensorRTGraphPool(assets, tmp_path, driver_tool=plan)
    assert pool.resident_state_enabled is expected
    clients = [pool.create_client(f"instance-{index}") for index in range(3)]
    # Later environment changes must not mix policies between lazy instances.
    monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", "0" if expected else "1")
    try:
        for client in clients:
            result = client.run("global_unified", {
                engine.key: np.array([], dtype=np.float32),
                engine.value: np.array([], dtype=np.float32),
                "token": np.array(1, dtype=np.float32), "scale": np.array(2, dtype=np.float32),
            })
            model = pool.instance_sessions[client.instance_id]["global_unified"]
            assert model._resident_enabled is expected
            assert isinstance(result[engine.out_key], runner.DeviceTensor if expected else np.ndarray)
            model.configure_profiling("global_unified", enabled=True, resident_state=not expected)
            assert model._resident_enabled is expected
    finally:
        drivers = list(pool.instance_cudas.values())
        pool.close()
    assert all(not driver.buffers for driver in drivers)
