"""Bounded Codex worker transport for core StructuredProposer.

The deployment-owned worker must enforce Codex isolation and its model binding.
This adapter freezes that worker/configuration, owns its process lifetime and
returns untrusted JSON to core's edit contract. It does not admit candidates.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import time

from PhysicalRSI_core.infra import processes, proposal_model, storage
from PhysicalRSI_core.infra.storage import atomic_json, canonical, file_digest, strict_json


class CodexProposal:
    def __init__(self, *, worker, deployment, scratch, python=sys.executable):
        self.worker = Path(worker).resolve(strict=True)
        self.deployment = Path(deployment).resolve(strict=True)
        self.python = Path(python).resolve(strict=True)
        self.scratch = Path(scratch).resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(kind='robotworld-codex-json-proposal/v1',
                    files={str(p): file_digest(p) for p in (
                        self.worker, self.deployment, self.python,
                        Path(__file__), Path(processes.__file__), Path(storage.__file__),
                        Path(proposal_model.__file__), Path(__file__).with_name('isolation.py'))})

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError('Codex proposal worker or deployment changed')

    def generate(self, request, limits):
        self._stable()
        if len(canonical(request)) > limits.request_bytes:
            raise ValueError('Proposal request exceeds byte allowance')
        with tempfile.TemporaryDirectory(prefix='codex-proposal-', dir=self.scratch) as folder:
            root = Path(folder)
            atomic_json(root / 'input.json', dict(request=request, limits=asdict(limits)))
            output = root / 'output.json'
            worker = processes.ManagedProcess('robotworld-codex-proposal', [
                str(self.python), str(self.worker), '--deployment', str(self.deployment),
                '--input', str(root / 'input.json'), '--output', str(output)], cwd=root,
                env_overrides={'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
                               'PYTHONDONTWRITEBYTECODE': '1'})
            started = time.monotonic()
            worker.start()
            try:
                while worker.poll() is None:
                    remaining = limits.seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError('Codex proposal total deadline reached')
                    time.sleep(min(.02, remaining))
                if time.monotonic() - started >= limits.seconds:
                    raise TimeoutError('Codex proposal total deadline reached')
                if worker.poll() != 0 or not output.is_file() or output.is_symlink():
                    raise RuntimeError('Codex proposal worker has no completed reply')
                if output.stat().st_size > limits.response_bytes:
                    raise ValueError('Codex proposal response exceeds byte allowance')
                reply = strict_json(output.read_text())
                if (not isinstance(reply, dict) or set(reply) != {'text', 'usage', 'provider', 'tool_calls'}
                        or not isinstance(reply['text'], str) or not isinstance(reply['tool_calls'], list)):
                    raise ValueError('Invalid Codex proposal reply envelope')
                self._stable()
                return reply
            finally:
                # Local teardown is bounded; provider cancellation/billing is not implied.
                worker.stop(timeout=1)
