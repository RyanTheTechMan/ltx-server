"""Instance-local cache and callback adapters for the pinned upstream API."""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


class RetainedModel:
    """Keep real weights, not merely an upstream meta-device model shell.

    A single inference thread owns this object. CPU parking preserves FP8 dtype;
    it never converts weights back to BF16 or reloads the checkpoint.
    """

    def __init__(self, build: Callable[..., Any], device: Any, *, park_on_exit: bool) -> None:
        self.build = build
        self.device = device
        self.park_on_exit = park_on_exit
        self.model: Any = None
        self.on_gpu = False

    def load(self, **kwargs: Any) -> Any:
        if self.model is None:
            self.model = self.build(**kwargs)
            self.on_gpu = True
        elif not self.on_gpu:
            self.model.to(self.device)
            self.on_gpu = True
        return self.model

    @contextmanager
    def context(self, **kwargs: Any) -> Iterator[Any]:
        model = self.load(**kwargs)
        try:
            yield model
        finally:
            # On failure the owner discards the runtime; a transfer must not mask an OOM.
            if self.park_on_exit and sys.exception() is None:
                self.park()

    def park(self) -> None:
        if self.model is not None and self.on_gpu:
            self.model.to("cpu")
            self.on_gpu = False

    def clear(self) -> None:
        # Release references; do not mutate cached model shells to meta while a call uses them.
        self.model = None
        self.on_gpu = False


class CallAdapter:
    def __init__(
        self, target: Any, before: Callable[[], None], after: Callable[[], None] | None = None
    ) -> None:
        self.target, self.before, self.after = target, before, after

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.before()
        try:
            return self.target(*args, **kwargs)
        finally:
            if self.after:
                self.after()
