import json
import pytest

from PhysicalRSI_baselines.robotworld.codex_proposals import CodexProposal
from PhysicalRSI_core.self_harness.proposals import ProposalLimits


def adapter(tmp_path, body):
    script = tmp_path / 'worker.py'
    script.write_text('import sys,json,time\nfrom pathlib import Path\n' + body)
    deployment = tmp_path / 'deployment.json'
    deployment.write_text('{}')
    return CodexProposal(worker=script, deployment=deployment, scratch=tmp_path / 'scratch')


def test_worker_reply_is_untrusted_data_not_a_candidate(tmp_path):
    reply = dict(text='{"candidates": []}', usage=None, provider={'model': 'fixture'}, tool_calls=[])
    transport = adapter(tmp_path, 'Path(sys.argv[sys.argv.index("--output")+1]).write_text(' + repr(json.dumps(reply)) + ')\n')
    assert transport.generate({'development': 'fixture'}, ProposalLimits()) == reply


def test_changed_worker_cannot_run_under_frozen_identity(tmp_path):
    transport = adapter(tmp_path, 'raise RuntimeError("must not execute")\n')
    transport.worker.write_text('pass\n')
    with pytest.raises(ValueError, match='changed'):
        transport.generate({}, ProposalLimits())


def test_worker_timeout_is_not_an_empty_proposal(tmp_path):
    transport = adapter(tmp_path, 'time.sleep(30)\n')
    with pytest.raises(TimeoutError):
        transport.generate({}, ProposalLimits(seconds=.1))


def test_oversized_reply_is_rejected(tmp_path):
    transport = adapter(tmp_path, 'Path(sys.argv[sys.argv.index("--output")+1]).write_text("x"*4096)\n')
    with pytest.raises(ValueError, match='response exceeds'):
        transport.generate({}, ProposalLimits(response_bytes=1024))
