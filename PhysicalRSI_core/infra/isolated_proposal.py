"""Untrusted System 2 Python strategies return data for normal schema admission."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
import uuid

from .isolated_program import PythonIsolation
from .storage import canonical, digest, file_digest


_RUNNER = r'''
import json
inputs = json.load(open('/input.json'))
scope = {}
exec(compile(inputs['source'], '<isolated-system2>', 'exec'), scope)
reply = scope['generate'](inputs['request'], inputs['limits'])
with open('/output/result.json', 'w') as stream:
    json.dump(dict(kind='result', value=reply), stream, allow_nan=False)
'''


class IsolatedProposal:
    def __init__(self, source, *, isolation, output):
        if not isinstance(source, str) or not source.strip() or not isinstance(isolation, PythonIsolation):
            raise ValueError("Declare System 2 source and a PythonIsolation profile")
        self.source, self.isolation, self.output = source, isolation, Path(output).resolve()
        self._frozen = deepcopy(self.identity())

    def identity(self):
        return dict(kind="isolated-system2-proposal/v1", source_sha256=digest(self.source),
                    isolation=self.isolation.identity(), implementation=file_digest(Path(__file__)))

    def generate(self, request, limits):
        if self.identity() != self._frozen:
            raise ValueError("Frozen isolated proposal changed")
        deadline = monotonic() + limits.seconds
        inputs = dict(source=self.source, request=request, limits=asdict(limits))
        if len(canonical(request)) > limits.request_bytes or len(canonical(inputs)) > self.isolation.message_bytes:
            raise ValueError("Isolated proposal request exceeds the byte allowance")
        result = self.isolation.run(_RUNNER, json.loads(canonical(inputs)),
                                    output=self.output / uuid.uuid4().hex, deadline=deadline)
        if len(canonical(result)) > limits.response_bytes:
            raise ValueError("Isolated proposal response exceeds the byte allowance")
        return result
