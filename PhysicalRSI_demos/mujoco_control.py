"""Headless MuJoCo fixtures through the shared controller and experiment ports.

Two distinct joint/actuator contracts exercise force and torque control. This is
a small dynamics integration, not a robot benchmark or physical qualification.
"""

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import Dependency, Embodiment, Revision, System1, System2Trial
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra.controller import ControllerGateway, GatewayEnvironment
from PhysicalRSI_core.infra.controller_rpc import ControllerClient, ControllerHost, controller_transport
from PhysicalRSI_core.infra.devices import DeviceRegistry
from PhysicalRSI_core.infra.services import ServiceSpec, Services
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, locked, read_json
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.experiments import experiment_result
from PhysicalRSI_core.self_harness.selection import select_survivor
from PhysicalRSI_core.timing import ControlTiming, SensorSample, action_chunk

SCOPE = "MuJoCo joint fixtures; simulation evidence only; no physical qualification"
TIMING = ControlTiming(.02, 5, 2, .25)
GAINS = {"slider": (10., 5., 10.), "hinge": (2., .5, 3.)}


def specification(model):
    if model not in GAINS:
        raise ValueError("Unknown MuJoCo fixture")
    name = "mujoco-" + model
    state_units, action_units = ("m,m/s", "N") if model == "slider" else ("rad,rad/s", "N*m")
    return Embodiment(name, file_digest(Path(__file__).with_name("mujoco_models") / (model + ".xml")),
                      "simulation", Contract("joint-state-v1", state_units, "joint", name),
                      Contract("joint-effort-v1", action_units, "joint", name),
                      resources=("simulation:" + model,), timing=TIMING)


class MuJoCoDriver:
    def __init__(self, model):
        import mujoco
        self.mujoco = mujoco
        self.fixture = model
        self.path = Path(__file__).with_name("mujoco_models") / (model + ".xml")
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.sequence = -1
        self.case = None

    def identity(self):
        return dict(kind="mujoco-joint-fixture", mujoco=self.mujoco.__version__, fixture=self.fixture,
                    model_sha256=file_digest(self.path), implementation=file_digest(Path(__file__)))

    def reset(self, case, context):
        context.check()
        for key in ("initial", "target", "position_tolerance", "velocity_tolerance"):
            value = case.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("MuJoCo case requires finite initial/target/tolerances")
        if (not -1 <= case["initial"] <= 1 or not -1 <= case["target"] <= 1
                or case["position_tolerance"] <= 0 or case["velocity_tolerance"] <= 0):
            raise ValueError("MuJoCo case exceeds the declared joint limits")
        self.case = dict(case)
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = case["initial"]
        self.mujoco.mj_forward(self.model, self.data)
        return dict(ready=True, terminated=self._reached())

    def _reached(self):
        return (abs(float(self.data.qpos[0]) - self.case["target"]) <= self.case["position_tolerance"]
                and abs(float(self.data.qvel[0])) <= self.case["velocity_tolerance"])

    def observe(self, context):
        context.check()
        self.sequence += 1
        # Copy native arrays; MuJoCo mutates its state buffers in place.
        payload = dict(qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist(),
                       simulation_seconds=float(self.data.time))
        return SensorSample(payload, self.sequence, monotonic())

    def validate(self, actions):
        low, high = self.model.actuator_ctrlrange[0]
        for action in actions:
            if (not isinstance(action, list) or len(action) != 1 or isinstance(action[0], bool)
                    or not isinstance(action[0], (int, float)) or not math.isfinite(action[0])
                    or not low <= action[0] <= high):
                raise ValueError("Expected one finite effort within the MuJoCo actuator limits")

    def execute(self, actions, *, period_seconds, deadline, context):
        self.validate(actions)
        substeps = round(period_seconds / self.model.opt.timestep)
        if substeps < 1 or not math.isclose(substeps * self.model.opt.timestep, period_seconds, abs_tol=1e-12):
            raise ValueError("Control period must be a multiple of the MuJoCo timestep")
        for index, action in enumerate(actions):
            self.data.ctrl[:] = action
            for _ in range(substeps):
                context.check()
                if monotonic() >= deadline:
                    raise TimeoutError("MuJoCo execution exceeded the controller deadline")
                self.mujoco.mj_step(self.model, self.data)
            if self._reached():
                return dict(terminated=True, executed_steps=index + 1)
        return dict(terminated=False, executed_steps=len(actions))

    def quiesce(self, context):
        context.check()
        self.data.ctrl[:] = 0
        return dict(quiescent=True, reason="Headless dynamics advance only during execute calls",
                    simulation_seconds=float(self.data.time))


def policy_revision(mode):
    return digest(dict(mode=mode, gains=GAINS, implementation=file_digest(Path(__file__))))


class JointPolicy:
    def __init__(self, model, mode):
        if mode not in {"zero-effort", "feedback"}:
            raise ValueError("Unknown controller mode")
        self.model, self.mode = model, mode

    def identity(self):
        return dict(name=self.mode, route=self.model, revision=policy_revision(self.mode))

    def describe(self):
        spec = specification(self.model)
        return System1(self.mode, policy_revision(self.mode), spec.observation, spec.action,
                       dependencies=(Dependency("joint-controller", file_digest(Path(__file__)), "skill"),),
                       timing=TIMING)

    def begin_episode(self, task, case, observation, context):
        self.target = case["target"]

    def act(self, observation, context):
        state = observation["payload"]
        kp, kd, limit = GAINS[self.model]
        effort = (max(-limit, min(limit, kp * (self.target - state["qpos"][0]) - kd * state["qvel"][0]))
                  if self.mode == "feedback" else 0.)
        return action_chunk(observation, [[effort], [effort]], period_seconds=TIMING.period_seconds)


class JointVerifier:
    def identity(self):
        return dict(name="independent-joint-state-check", revision=file_digest(Path(__file__)))

    def verify(self, case, trace):
        state = trace[-1]["observation"]["payload"]
        error, speed = abs(state["qpos"][0] - case["target"]), abs(state["qvel"][0])
        return dict(outcome="success" if error <= case["position_tolerance"] and speed <= case["velocity_tolerance"]
                    else "failure", reason="Compare measured simulator state with the declared target and tolerances",
                    measurements=dict(position_error=error, speed=speed, simulation_seconds=state["simulation_seconds"]))


def make_gateway(root, model):
    return ControllerGateway(root, driver=MuJoCoDriver(model), embodiment=specification(model))


@contextmanager
def controller_environments(root):
    """Own two separate controller processes for a comparison or campaign."""
    root = Path(root).resolve()
    specs = []
    for model in GAINS:
        directory = root / "controllers" / model
        probe = make_gateway(directory, model)
        try:
            expected = probe.identity()
        finally:
            probe.close()
        specs.append(ServiceSpec(model, expected, command=(sys.executable, "-m",
            "PhysicalRSI_demos.mujoco_control", "--serve", model, "--workspace", str(directory))))
    with Services(root / "services", client_factory=controller_transport) as services:
        clients = services.start(specs)
        yield {spec.name: GatewayEnvironment(ControllerClient(clients[spec.name], expected=spec.expected))
               for spec in specs}


def run(workspace):
    root = Path(workspace).resolve()
    with locked(root / ".comparison.lock"):
        return _run(root)


def _run(root):
    candidates = {mode: policy_revision(mode) for mode in ("zero-effort", "feedback")}
    comparison_path = root / "comparison.json"
    if comparison_path.exists():
        comparison, cohort = read_json(comparison_path), read_json(root / "cohort.json")
        if comparison["candidates"] != candidates:
            raise ValueError("Changed controller implementation requires a fresh comparison workspace")
    else:
        comparison = dict(schema_version=1, round_id="mujoco-control", parent_id="zero-effort",
                          frozen_at=now(), candidates=candidates, scope=SCOPE,
                          evaluation_kind="simulation_evaluation", protocol_sha256=digest("joint-target-v1"),
                          profile=dict(tasks={model: dict(weight=1, episodes=2, score_range=[0, 1], maximum_regression=0)
                                              for model in GAINS}, minimum_gain=0, tie_tolerance=1e-10))
        cases = {model: [dict(embodiment=model, initial=initial, target=target,
                             position_tolerance=.02, velocity_tolerance=.05)
                         for initial, target in ((0., .35), (-.1, .2))] for model in GAINS}
        atomic_json(comparison_path, comparison)
        cohort = dict(split="validation", generated_at=now(), comparison_sha256=digest(comparison),
                      layouts={model: [digest(case) for case in values] for model, values in cases.items()}, cases=cases)
        atomic_json(root / "cohort.json", cohort)
    runtime = ExperimentRuntime(root / "trials", device_registry=DeviceRegistry(root / "devices"))
    results = []
    with controller_environments(root) as environments:
        for mode, revision in candidates.items():
            run_ids = []
            origin = System2Trial(Revision("declared-controller-comparison", "1"), mode, revision, "validation",
                                  digest(comparison), digest(cohort))
            for model, cases in cohort["cases"].items():
                for index, case in enumerate(cases):
                    run_id = f"{mode}-{model}-{index}"
                    runtime.run(run_id, task=model, case=case, scope=SCOPE, environment=environments[model],
                                policy=JointPolicy(model, mode), verifier=JointVerifier(),
                                budget=Budget(max_steps=100, seconds=30), system2=origin)
                    run_ids.append(run_id)
            results.append(experiment_result(runtime, run_ids, candidate_id=mode, comparison=comparison,
                                             cohort=cohort, evaluator_revision="joint-target-v1", evidence_root=root))
    decision = select_survivor(comparison, cohort, results, evidence_root=root)
    result = dict(selection=decision, results=results, scope=SCOPE, qualification=None, lineage_committed=False)
    atomic_json(root / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--serve", choices=tuple(GAINS))
    args = parser.parse_args()
    if args.serve:
        ControllerHost(make_gateway(args.workspace, args.serve)).serve(
            transport="http", host="127.0.0.1", port=0, parent_watch=True)
    else:
        print(json.dumps(run(args.workspace), indent=2))


if __name__ == "__main__":
    main()
