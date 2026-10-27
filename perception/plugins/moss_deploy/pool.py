"""Shared immutable graph weights with per-runtime cache/context ownership."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .assets import ModelAssets


@dataclass(frozen=True)
class OutputName:
    name: str


class GraphClient:
    def __init__(self, pool: "SharedGraphPool", instance_id: str, runners: dict[str, Callable]):
        self.pool = pool
        self.instance_id = instance_id
        self.runners = runners

    def run(self, name: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.runners[name](feeds)


class SharedGraphPool:
    """Interface for shared engines and distinct instance execution contexts.

    A custom backend supplies context_factory(instance_id)->graph-name->runner.
    Each runner accepts named NumPy inputs and returns named NumPy outputs.
    No KV/audio/RNG state is stored in this pool. Immutable reference codes
    can be shared; caller receives its own mutable list representation.
    """
    def __init__(self, assets: ModelAssets, context_factory: Callable[[str], dict[str, Callable]], *, backend: str):
        self.assets = assets
        self.context_factory = context_factory
        self.backend = backend
        self._instance_ids = set()
        self._instance_lock = threading.Lock()
        self.reference_codes: dict[tuple, tuple[tuple[int, ...], ...]] = {}
        self.reference_lock = threading.Lock()

    def create_client(self, instance_id: str) -> GraphClient:
        with self._instance_lock:
            if instance_id in self._instance_ids:
                raise ValueError(f"Instance ID already owns contexts: {instance_id}")
            runners = self.context_factory(instance_id)
            missing = set(self.assets.graph_paths) - set(runners)
            if missing:
                raise ValueError(f"Backend context factory did not supply graphs: {sorted(missing)}")
            self._instance_ids.add(instance_id)
        return GraphClient(self, instance_id, runners)


class OnnxGraphPool(SharedGraphPool):
    """One CPU ORT session per graph, shared by independent request contexts."""
    def __init__(self, assets: ModelAssets, *, thread_count: int = 2):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(thread_count))
        options.inter_op_num_threads = 1
        self.sessions = {name: ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]) for name, path in assets.graph_paths.items()}
        def factory(instance_id):
            runners = {}
            for name, session in self.sessions.items():
                output_names = [output.name for output in session.get_outputs()]
                def run(feeds, session=session, names=output_names):
                    return dict(zip(names, session.run(None, feeds), strict=True))
                runners[name] = run
            return runners
        super().__init__(assets, factory, backend="onnx-cpu")


class SessionAdapter:
    """Small original-runtime ORT API bridge, without keeping shared caches."""
    def __init__(self, client: GraphClient, name: str, output_names: list[str], transform: Callable | None = None):
        self.client, self.name, self.output_names, self.transform = client, name, output_names, transform

    def get_outputs(self):
        return [OutputName(name) for name in self.output_names]

    def run(self, output_names, feeds):
        if self.transform is None:
            outputs = self.client.run(self.name, feeds)
        else:
            outputs = self.transform(feeds)
        return [outputs[name] for name in (output_names or self.output_names)]
