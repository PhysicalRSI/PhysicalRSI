"""Optional Gymnasium adapters for the PhysicalRSI experiment ports.

Gymnasium is deliberately imported only when an adapter is constructed or
used.  Projects that do not use Gymnasium can therefore install and import
the public ``physicalrsi`` package without the optional dependency.
"""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from PhysicalRSI_core.contracts import Context, Contract, contract_from
from PhysicalRSI_core.embodiment import Embodiment, System1
from PhysicalRSI_core.infra.storage import digest


def _gymnasium():
    """Return Gymnasium or raise an actionable optional-dependency error."""
    try:
        import gymnasium
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "Gymnasium support requires the optional dependency. "
            "Install it with `pip install 'physicalrsi[gym]'`."
        ) from exc
    return gymnasium


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays and nested containers to JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    # Importing numpy here keeps it optional for users that never construct a
    # Gymnasium adapter.  Gymnasium itself depends on numpy at runtime.
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Gymnasium observations should be JSON-compatible after the numpy cases
    # above.  Let the caller see a useful error for unsupported custom values.
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Gymnasium value is not JSON-compatible: {type(value)!r}") from exc


def _from_json(value: Any, space: Any = None) -> Any:
    """Convert a JSON action back to the numpy/scalar type expected by a space."""
    if space is None:
        return value
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - Gymnasium always supplies numpy
        return value

    name = type(space).__name__
    if name == "Discrete":
        return int(value)
    if name in {"Box", "MultiBinary", "MultiDiscrete"}:
        dtype = getattr(space, "dtype", None)
        array = np.asarray(value, dtype=dtype)
        # Gymnasium accepts scalar numpy values for scalar-shaped spaces, while
        # preserving arrays for vector and matrix actions.
        return array.item() if getattr(array, "shape", ()) == () else array
    if isinstance(value, dict):
        spaces = getattr(space, "spaces", {})
        return {key: _from_json(item, spaces.get(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        spaces = getattr(space, "spaces", ())
        return type(value)(_from_json(item, subspace) for item, subspace in zip(value, spaces))
    return value


def _callable_identity(function: Callable | None) -> dict[str, str] | None:
    if function is None:
        return None
    return {
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
    }


def _invoke(function: Callable, case: dict, trace: list[dict]):
    """Support verifier callbacks accepting (case, trace), trace, or no args."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(case, trace)
    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return function(case, trace)
    positional = [parameter for parameter in parameters if parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
    )]
    if len(positional) >= 2:
        return function(case, trace)
    if len(positional) == 1:
        return function(trace)
    return function()


class GymnasiumEnvironment:
    """Expose any ``gymnasium.Env`` through the core Environment port."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        *,
        observation_contract: Contract | dict,
        action_contract: Contract | dict,
        embodiment_name: str,
        mode: str = "simulation",
        max_steps: int,
    ):
        if not callable(env_factory):
            raise ValueError("env_factory must be callable")
        if not isinstance(embodiment_name, str) or not embodiment_name.strip():
            raise ValueError("embodiment_name must be a nonempty string")
        if mode not in {"software", "simulation", "physical"}:
            raise ValueError("mode must be software, simulation, or physical")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.env_factory = env_factory
        self.observation_contract = contract_from(observation_contract)
        self.action_contract = contract_from(action_contract)
        self.embodiment_name = embodiment_name
        self.mode = mode
        self.max_steps = max_steps
        self._env = None
        self._steps = 0

    def identity(self) -> dict:
        return {
            "kind": "physicalrsi.gymnasium-environment/v1",
            "factory": _callable_identity(self.env_factory),
            "embodiment": self.embodiment_name,
            "mode": self.mode,
            "max_steps": self.max_steps,
            "observation_contract": self.observation_contract.__dict__,
            "action_contract": self.action_contract.__dict__,
        }

    def describe(self) -> Embodiment:
        return Embodiment(
            self.embodiment_name,
            digest(self.identity()),
            self.mode,
            self.observation_contract,
            self.action_contract,
        )

    def _require_env(self):
        _gymnasium()
        if self._env is None:
            self._env = self.env_factory()
            if not callable(getattr(self._env, "reset", None)) or not callable(getattr(self._env, "step", None)):
                raise TypeError("env_factory must return a Gymnasium environment")
        return self._env

    def reset(self, case: dict, context: Context) -> dict:
        context.check()
        if not isinstance(case, dict) or "seed" not in case:
            raise ValueError("Gymnasium reset cases require a seed")
        env = self._require_env()
        result = env.reset(seed=case["seed"])
        observation = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        self._steps = 0
        context.check()
        return {"ready": True, "observation": _json_value(observation), "terminated": False}

    def step(self, action, context: Context) -> dict:
        context.check()
        env = self._require_env()
        if self._steps >= self.max_steps:
            raise RuntimeError("Gymnasium environment reached max_steps; reset before stepping")
        native_action = _from_json(action, getattr(env, "action_space", None))
        result = env.step(native_action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("Gymnasium Env.step must return observation, reward, terminated, truncated, info")
        observation, _reward, terminated, truncated, _info = result
        self._steps += 1
        done = bool(terminated or truncated or self._steps >= self.max_steps)
        context.check()
        return {"observation": _json_value(observation), "terminated": done}

    def close(self):
        if self._env is not None and callable(getattr(self._env, "close", None)):
            self._env.close()


class GymnasiumVerifier:
    """Adapt a success predicate and optional measurement callback to Verifier."""

    def __init__(self, success_fn: Callable, measurements_fn: Callable | None = None):
        if not callable(success_fn):
            raise ValueError("success_fn must be callable")
        if measurements_fn is not None and not callable(measurements_fn):
            raise ValueError("measurements_fn must be callable")
        self.success_fn = success_fn
        self.measurements_fn = measurements_fn

    def identity(self) -> dict:
        return {
            "kind": "physicalrsi.gymnasium-verifier/v1",
            "success": _callable_identity(self.success_fn),
            "measurements": _callable_identity(self.measurements_fn),
        }

    def verify(self, case: dict, trace: list[dict]) -> dict:
        result = _invoke(self.success_fn, case, trace)
        if isinstance(result, dict):
            outcome = result.get("outcome")
            if outcome not in {"success", "failure", "uncertain"}:
                raise ValueError("success_fn result dict must declare a valid outcome")
        elif isinstance(result, str) and result in {"success", "failure", "uncertain"}:
            outcome = result
        else:
            outcome = "success" if bool(result) else "failure"
        measurements = _invoke(self.measurements_fn, case, trace) if self.measurements_fn else {}
        if not isinstance(measurements, dict):
            raise ValueError("measurements_fn must return a JSON object")
        if isinstance(result, dict) and result.get("measurements"):
            measurements = {**result["measurements"], **measurements}
        return {"outcome": outcome, "reason": "Gymnasium success predicate", "measurements": _json_value(measurements)}


class CallablePolicy:
    """Load ``control.json`` parameters and call ``policy_fn(params, observation)``."""

    def __init__(
        self,
        candidate_or_policy: dict | str | Path | Callable,
        policy_or_candidate: Callable | dict | str | Path | None = None,
        *,
        control_file: str = "control.json",
        observation_contract: Contract | dict | None = None,
        action_contract: Contract | dict | None = None,
        name: str = "callable-policy",
    ):
        if callable(candidate_or_policy):
            self.policy_fn = candidate_or_policy
            candidate = policy_or_candidate
        else:
            candidate = candidate_or_policy
            self.policy_fn = policy_or_candidate
        if not callable(self.policy_fn):
            raise ValueError("CallablePolicy requires a policy function")
        if not isinstance(control_file, str) or not control_file or Path(control_file).name != control_file:
            raise ValueError("control_file must be a simple file name")
        self.control_file = control_file
        self.name = name
        self._candidate = candidate
        self.params = self._load_params(candidate)
        self.observation_contract = contract_from(observation_contract) if observation_contract is not None else None
        self.action_contract = contract_from(action_contract) if action_contract is not None else None
        if (self.observation_contract is None) != (self.action_contract is None):
            raise ValueError("Declare both observation_contract and action_contract")

    def _load_params(self, candidate):
        if candidate is None:
            return {}
        if isinstance(candidate, (str, Path)):
            path = Path(candidate) / self.control_file
            return json.loads(path.read_text())
        if isinstance(candidate, dict) and "root" in candidate:
            return json.loads((Path(candidate["root"]) / self.control_file).read_text())
        if isinstance(candidate, dict):
            return deepcopy(candidate)
        raise TypeError("candidate must be a harness declaration, path, or parameter mapping")

    def _candidate_revision(self):
        # The runtime calls describe() on every control step; verifying the
        # harness bytes each time made trials I/O-bound. The candidate is
        # verified once when first described and the revision is reused; the
        # Self-Harness re-verifies every harness before and after evaluation.
        # A harness that fails verification is an error, never a silent fallback.
        cached = getattr(self, "_revision_cache", None)
        if cached is not None:
            return cached
        if isinstance(self._candidate, dict) and {"root", "components", "id"} <= self._candidate.keys():
            from PhysicalRSI_core.self_harness.artifacts import verify_harness
            revision = verify_harness(self._candidate)
        else:
            revision = digest(self.params)
        self._revision_cache = revision
        return revision

    def identity(self) -> dict:
        return {
            "kind": "physicalrsi.callable-policy/v1",
            "name": self.name,
            "function": _callable_identity(self.policy_fn),
            "control_file": self.control_file,
            "parameters_sha256": sha256(json.dumps(self.params, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "revision": self._candidate_revision(),
        }

    def describe(self) -> System1:
        if self.observation_contract is None:
            raise ValueError("CallablePolicy.describe requires observation and action contracts")
        return System1(self.name, self._candidate_revision(), self.observation_contract, self.action_contract)

    def begin_episode(self, task: str, case: dict, observation, context: Context):
        return None

    def act(self, observation, context: Context):
        context.check()
        return _json_value(self.policy_fn(deepcopy(self.params), deepcopy(observation)))

    def end_episode(self, context: Context):
        return {"stopped": True}


__all__ = ["GymnasiumEnvironment", "GymnasiumVerifier", "CallablePolicy"]
