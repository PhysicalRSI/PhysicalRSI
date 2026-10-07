"""Native LIBERO primitives with a read-only public RGB-D object locator."""
import math
import time

from .libero_primitives import NativeLiberoPrimitives


class LocatedLiberoPrimitives(NativeLiberoPrimitives):
    def __init__(self, *args, object_locator, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_locator = object_locator

    @property
    def handlers(self):
        handlers = super().handlers

        def locate(args, kwargs, *, deadline):
            if not math.isfinite(deadline):
                raise ValueError("A finite perception deadline is required")
            if time.monotonic() >= deadline:
                raise TimeoutError("Object-location deadline reached")
            result = self._object_locator(
                self.get_observation(), *args, **kwargs, deadline=deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError("Object-location deadline reached")
            return result

        handlers["locate_object"] = locate
        return handlers
