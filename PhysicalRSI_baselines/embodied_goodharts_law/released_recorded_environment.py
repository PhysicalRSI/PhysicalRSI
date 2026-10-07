"""Drop simulator-owning callbacks after successful native teardown."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest
from .recorded_cap import RecordedPrimitiveEnvironment


class ReleasedRecordedEnvironment(RecordedPrimitiveEnvironment):
    def identity(self):
        return {**super().identity(), "release_implementation_sha256": file_digest(Path(__file__))}

    def close(self):
        # Failed teardown remains an error with its references retained for
        # diagnosis; this cannot certify quiescence or release a device lease.
        if getattr(self, "_teardown_error", None) is not None:
            raise self._teardown_error
        try:
            super().close()
        except BaseException as exc:
            self._teardown_error = exc
            raise
        self._finished = True
        self._handlers = {}
        self._measure = None
        self._api = None
