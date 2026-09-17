"""Instance-local serialization for Pi0.5 mutable GPU resources."""
from functools import wraps


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lifecycle_lock:
            if self._reload_failed:
                raise RuntimeError("Pi0.5 weight reload failed after mutation; construct a new frontend")
            return method(self, *args, **kwargs)
    return call


def reload_guard(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        self._reload_mutating = False
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            if self._reload_mutating:
                self._reload_failed = True
            raise
        finally:
            self._reload_mutating = False
    return serialized(call)
