"""Retain robot-only planner status without changing planning or control."""


class RobotPlannerDiagnostics:
    def __init__(self, planner):
        if planner.world_config is not None:
            raise ValueError('Only robot-only planners are supported')
        self._planner = planner

    @property
    def world_config(self):
        return self._planner.world_config

    def plan_path(self, *args, **kwargs):
        if self.world_config is not None:
            raise ValueError('Planner acquired a scene map')
        original = self._planner.motion_gen
        status = []

        class MotionCapture:
            def __getattr__(self, name):
                return getattr(original, name)

            def plan_single(self, *call_args, **call_kwargs):
                result = original.plan_single(*call_args, **call_kwargs)
                value = getattr(result, 'status', None)
                status.append(None if value is None else str(value))
                return result

        self._planner.motion_gen = MotionCapture()
        try:
            result = self._planner.plan_path(*args, **kwargs)
            return dict(result, robot_planner_statuses=status,
                        diagnostic_scope='robot-only planner result status')
        finally:
            self._planner.motion_gen = original
