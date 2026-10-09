"""Reuse upstream event fixtures to check adapter network policy, without API calls."""
from types import SimpleNamespace

import pytest
from hybrid_rollout.robodojo.test_network_continue import NET, Rollout, call, completed, event, worker
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_sim import DexJoCoProfile
from PhysicalRSI_demos.dexterous_manipulation.runtime_profile import RuntimeProfile


class AdapterRollout(Rollout):
    def handlers(self): return dict(robodojo_execute=self.execute)


def test_simulation_keeps_upstream_same_thread_recovery_without_action_replay(tmp_path,monkeypatch):
    controller=worker(tmp_path,monkeypatch,[event("error",error=NET,willRetry=False),
        completed(),call(),completed("t1","completed",None)])
    controller.runtime=SimpleNamespace(allow_same_thread_network_continue=DexJoCoProfile.allow_same_thread_network_continue,
                                       initial_prompt=lambda rollout:"Run this test fixture")
    rollout=AdapterRollout()
    controller.run(rollout)
    assert rollout.actions==1
    assert controller.transport.turns==2
    assert {p["threadId"] for m,p in controller.transport.requests if m=="turn/start"}=={"thread"}


def test_default_profile_requires_reconciliation_after_model_failure(tmp_path,monkeypatch):
    controller=worker(tmp_path,monkeypatch,[event("error",error=NET,willRetry=False),completed()])
    controller.runtime=SimpleNamespace(allow_same_thread_network_continue=RuntimeProfile.allow_same_thread_network_continue,
                                       initial_prompt=lambda rollout:"Run this test fixture")
    rollout=AdapterRollout()
    with pytest.raises(RuntimeError,match="reconcile"):
        controller.run(rollout)
    assert rollout.actions==0 and controller.transport.turns==1
