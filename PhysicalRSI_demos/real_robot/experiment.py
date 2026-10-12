"""Initial model-written L1 and explicit post-rescue source/memory development."""
from .contracts import CapabilityUnavailable
from .controllers import L1Controller
from .evidence import write_new


class DevelopingL1:
    def __init__(self, *, library, model, runtime, parent=None, perception=None, seed=None):
        self.library, self.model, self.runtime = library, model, runtime
        self.parent = parent
        self.perception = perception
        self.executions = self.improvements = 0
        self.seed = seed
        self.proposals = 0

    def _capabilities(self, context):
        available = dict(getattr(context.robot, "capabilities", {}),
                         look=self.perception is not None,
                         align=self.perception is not None,
                         relate=self.perception is not None)
        previous = [e for e in context.episode.events if e["kind"] == "execution_capabilities"]
        if not previous or previous[-1]["capabilities"] != available:
            context.episode.event("execution_capabilities", level="L1", capabilities=available)
        return available

    def run(self, context):
        capabilities = self._capabilities(context)
        if self.parent is None:
            self.proposals += 1
            obs = context.checkpoint(force=True)
            feedback = [dict(task=context.task.id, split="development",
                             instruction=context.task.public(), observations=[obs.public()],
                             execution=[], outcome="not_attempted", solved_by=None,
                             capabilities=capabilities)]
            try:
                self.parent = self.library.propose(task=context.task, parent=None, feedback=feedback,
                    model=self.model, output=context.episode.root / f"initial-proposal-{self.proposals:04d}",
                    seed=self.seed)
            except Exception as exc:
                raise CapabilityUnavailable("L1 proposal unavailable: " + type(exc).__name__) from exc
        try:
            program = self.library.load_for_execution(self.parent, foundation=self.library.foundation)
        except ValueError:
            # Source/memory may remain useful after adapter development. Preserve
            # the old freeze, explicitly create a new unvalidated candidate, and
            # never reuse the old comparison as proof for the changed system.
            previous = self.parent
            program = self.library.load(previous)
            self.parent = self.library.freeze(task=context.task, source=program["source"], memory=program["memory"])
            context.episode.event("l1_policy_rebased", level="L1", previous_id=previous["id"],
                                  candidate_id=self.parent["id"], promoted=False,
                                  reason="Adapter/API/foundation changed; existing validation does not transfer")
        context.episode.event("l1_policy_selected", level="L1", policy_id=self.parent["id"],
                              manifest=self.parent)
        self.executions += 1
        L1Controller(program, self.runtime, perception=self.perception,
                     runtime_name=f"l1-runtime-{self.executions:04d}").run(context)

    def improve(self, context, rescued_by):
        """Development checkpoint after L2 OR L3; never promote an online candidate."""
        from .usage import usage
        counts = usage(context.episode.events)["controller_entries"]
        feedback = context.episode.development_feedback(dict(outcome="uncertain", solved_by=None,
            l2_calls=counts["L2"], l3_calls=counts["L3"]))
        feedback["capabilities"] = self._capabilities(context)
        self.improvements += 1
        context.robot.hold()
        previous = self.parent
        try:
            candidate = self.library.propose(task=context.task, parent=previous, feedback=[feedback],
                model=self.model, output=context.episode.root / f"online-proposal-{self.improvements:04d}",
                seed=self.seed if previous is None else None)
        except Exception as exc:
            # Proposal parsing/model failures do not make a confirmed robot
            # action uncertain. Retain the input and continue the hierarchy.
            context.episode.event("online_learning_pending", level="L1", rescued_by=rescued_by.value,
                                  error_type=type(exc).__name__, promoted=False)
            return False
        context.episode.event("online_learning", level="L1", rescued_by=rescued_by.value,
            parent_id=previous["id"] if previous else None,
            candidate_id=candidate["id"] if candidate else None, promoted=False,
            validation_required=True)
        if candidate is not None:
            self.parent = candidate
            return True
        return False

    def learn(self, episode, result):
        if not result["l2_calls"] and not result["l3_calls"] and not result.get("usage", {}).get("l3_uses"):
            return None
        feedback = [episode.development_feedback(result)]
        declared = [e["capabilities"] for e in episode.events if e["kind"] == "execution_capabilities"]
        if declared:
            feedback[0]["capabilities"] = declared[-1]
        candidate = self.library.propose(task=episode.task, parent=self.parent, feedback=feedback,
            model=self.model, output=episode.root / "learn-from-rescue")
        record = dict(candidate=candidate, parent=self.parent, outcome=result["outcome"],
                      stage="candidate_requires_independent_validation" if candidate else "unchanged",
                      promoted=False, physical_success_improvement=None)
        write_new(episode.root / "learning.json", record)
        return record
