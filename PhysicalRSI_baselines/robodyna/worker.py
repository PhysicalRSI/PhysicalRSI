"""Python 3.10 native simulator worker. JSON RPC uses a dedicated output FD.

No Core imports: the parent process owns skills, research, budgets and lineage.
Task code and scoring are imported from a separately pinned upstream checkout.
"""
import argparse
from collections import deque
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


class NativeWorker:
    def __init__(self, upstream, output):
        self.root = Path(upstream).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        os.chdir(self.root)
        sys.path[:0] = [str(self.root), str(self.root/'script'), str(self.root/'script/bench_script')]
        self.env = None
        self.video = None
        self.events = (self.output/'events.jsonl').open('x')
        self.samples = deque(maxlen=32)
        self.sim_steps = 0
        self.frames = 0
        self.success_snapshot = None
        self.budget_score_snapshot = None
        self.execution_seconds = 0.
        self.max_sim_steps = 15000
        self.started = time.monotonic()

    def reset(self, request):
        import yaml
        from evaluation_protocol import configure_evaluation
        from execution import InterceptionExecutor
        self.task = request['task']
        if self.task not in ('catch_ramp_ball', 'catch_valley_ball'):
            raise ValueError('Unsupported task')
        args = yaml.safe_load((self.root/'task_config/demo_dynamic.yml').read_text())
        args.update(task_name=self.task, task_config='demo_dynamic', render_freq=0,
                    save_data=False, eval_mode=True, save_path=str(self.output/'data'))
        _, scenario, official_seeds = configure_evaluation(self.task, args, request['scenario'])
        robot = str(self.root/'assets/embodiments/ur5-wsg')
        cfg = yaml.safe_load((Path(robot)/'config.yml').read_text())
        args.update(left_robot_file=robot, right_robot_file=robot,
                    left_embodiment_config=cfg, right_embodiment_config=cfg,
                    dual_arm_embodied=False, embodiment_dis=.6)
        module = importlib.import_module('envs.'+self.task)
        self.env = getattr(module, self.task)()
        # Official interactive robot reset enables contact-driven box physics.
        # This differs from the published VLA evaluator and is recorded explicitly.
        self.env._interactive_robot_mode = True
        t0 = time.monotonic()
        self.env.setup_demo(now_ep_num=0, seed=request['seed'], is_test=True, **args)
        self.reset_seconds = time.monotonic()-t0
        self.rollout_started = time.monotonic()
        self.env.configure_async_inference(enabled=request['mode'] == 'async')
        self.mode = request['mode']
        self.executor = InterceptionExecutor(self.env, self.task)
        device = self.env.scene.render_system.device
        self.device = dict(name=device.name, pci_bus=device.pci_string, cuda_id=device.cuda_id)
        if device.cuda_id != 0:
            raise RuntimeError('Renderer is not using the assigned CUDA device')
        (self.output/'device.json').write_text(json.dumps(self.device, indent=2))
        self._install_observer()
        self._sample()
        self._record_frame()
        self._event('reset', dict(request=request, config=args, official_seeds=official_seeds,
            control_mode='native-interactive-robot', observation_mode='simulator-state-and-static-geometry',
            reset_seconds=self.reset_seconds))
        return self.observe()

    def _phase(self):
        native = getattr(self.env, '_ball_phase', None)
        return {'released':'flight', 'roll':'rolling', 'rolling':'rolling', 'drop':'dropping',
                'dropping':'dropping'}.get(native, 'held')

    def _sample(self):
        phase = self._phase()
        if self.samples and self.samples[-1]['phase'] != phase:
            self.samples.clear()
        sample = dict(timestamp_s=self.sim_steps*float(self.env.scene.get_timestep()),
                      position_m=self.env.ball.get_pose().p.tolist(), phase=phase)
        if not self.samples or sample['timestamp_s'] != self.samples[-1]['timestamp_s']:
            self.samples.append(sample)
            self._event('sample', sample)

    def _install_observer(self):
        original = self.env.scene.step
        self.env._exp_sim_steps = 0
        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            self.sim_steps += 1
            self.env._exp_sim_steps = self.sim_steps
            if self.success_snapshot is None and self.sim_steps <= self.max_sim_steps and self.env.check_success():
                self.success_snapshot = dict(success=True, progress=float(self.env.get_score()),
                    score_detail=self.env.get_score_detail(), sim_steps=self.sim_steps,
                    ball_position_m=self.env.ball.get_pose().p.tolist())
                self._event('native_success', self.success_snapshot)
            # Native actions may finish beyond this boundary. Retain the native
            # score at the budget, without claiming to interrupt physics.
            if self.sim_steps == self.max_sim_steps:
                self.budget_score_snapshot = dict(progress=float(self.env.get_score()),
                    score_detail=self.env.get_score_detail(), failure_reason=self.env.get_fail_reason())
            if self.sim_steps % 10 == 0:
                self._sample()
            if self.sim_steps % 25 == 0:
                self._record_frame()
            return result
        self.env.scene.step = step

    def _record_frame(self):
        import imageio.v2 as imageio
        obs = self.env.get_obs()
        pixels = obs['observation']['head_camera']['rgb']
        if self.video is None:
            self.video = imageio.get_writer(str(self.output/'episode.mp4'), fps=10,
                                           codec='libx264', macro_block_size=1)
        self.video.append_data(pixels)
        self.frames += 1

    def _event(self, name, value):
        self.events.write(json.dumps(dict(event=name, value=value), default=json_value, allow_nan=False)+'\n')
        self.events.flush()

    def observe(self):
        env = self.env
        self._sample()
        if self.task == 'catch_ramp_ball':
            path = [env._surface_point(0., env._y_top).tolist(), env._surface_point(0., env._y_edge).tolist()]
            catcher = env.cup
            capture_height = float(env.table_top+.075)
        else:
            path = [list(map(float, p)) for p in env._surface_ball_path]
            catcher = env.bowl
            capture_height = float(env.table_top+env.BOX_HEIGHT)
        target_origin = list(map(float, catcher.get_pose().p))
        target_origin[2] = capture_height
        policy_observation = dict(samples=list(self.samples), phase=self._phase(),
            geometry=dict(path_m=path, capture_height_m=capture_height, gravity_m_s2=9.81),
            robot=dict(target_position_m=target_origin, workspace_min_m=[-.5,-.6,float(env.table_top)],
                       workspace_max_m=[.5,.7,1.6], max_speed_m_s=.5, command_latency_s=0.))
        # Verdict and score never enter the shared skill's input.
        success = self.success_snapshot is not None or (self.sim_steps <= self.max_sim_steps and bool(env.check_success()))
        score_snapshot = self.success_snapshot or self.budget_score_snapshot
        progress = score_snapshot['progress'] if score_snapshot else float(env.get_score())
        native = dict(success=success, progress=progress, score_detail=(score_snapshot['score_detail'] if score_snapshot else env.get_score_detail()),
                      failure_reason=('' if success else self.budget_score_snapshot['failure_reason'] if self.budget_score_snapshot else env.get_fail_reason()), sim_steps=self.sim_steps,
                      sim_time_s=self.sim_steps*float(env.scene.get_timestep()),
                      latency=env.async_inference_stats(), frames=self.frames,
                      executor_phase=str(getattr(self.executor, 'phase', 'unknown')),
                      episode_wall_seconds=time.monotonic()-self.rollout_started,
                      execution_wall_seconds=self.execution_seconds,
                      adapter_max_sim_steps=self.max_sim_steps,
                      latency_scope='shared_skill_inference_only')
        terminated = success or self.sim_steps >= self.max_sim_steps or bool(env._episode_past_deadline())
        native['stop_reason'] = 'native_success' if success else 'adapter_simulation_budget' if self.sim_steps >= self.max_sim_steps else 'native_deadline' if terminated else None
        return dict(policy_observation=policy_observation, native=native, terminated=terminated)

    def step(self, action):
        self._event('action', action)
        latency = float(action.get('inference_seconds', 0.))
        if not math.isfinite(latency) or latency < 0:
            raise ValueError('Invalid inference latency')
        env = self.env
        env._inference_calls += 1
        env._inference_time_total += latency
        if self.mode == 'async':
            env.advance_idle_steps(round(latency*env._async_sim_hz))
        if self.success_snapshot is None and self.sim_steps < self.max_sim_steps and not env._episode_past_deadline():
            if action['kind'] == 'observe':
                env.advance_idle_steps(int(action.get('steps', 25)))
            elif action['kind'] == 'target':
                before = self.sim_steps
                execution_start = time.monotonic()
                execution = self.executor.execute_target(action['target_position_m'])
                self.execution_seconds += time.monotonic()-execution_start
                self._event('execution', execution)
                if self.sim_steps == before:
                    env.advance_idle_steps(25)
            elif action['kind'] == 'qpos':
                env.take_action(action['positions'], action_type='qpos')
            else:
                raise ValueError('Unknown action kind')
        result = self.observe()
        self._event('verdict', result['native'])
        return result

    def close(self):
        if self.video is not None:
            self.video.close()
            self.video = None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        summary = dict(wall_seconds=time.monotonic()-self.started, user_cpu_seconds=usage.ru_utime,
                       system_cpu_seconds=usage.ru_stime, max_rss_kib=usage.ru_maxrss,
                       frames=self.frames, sim_steps=self.sim_steps,
                       initialization_seconds=getattr(self, 'reset_seconds', None),
                       device=getattr(self, 'device', None))
        (self.output/'resources.json').write_text(json.dumps(summary, indent=2))
        self.events.close()
        if self.env is not None:
            self.env.close_env()
            self.env = None
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reply-fd', type=int, required=True)
    args = parser.parse_args()
    reply = os.fdopen(args.reply_fd, 'w', buffering=1)
    worker = NativeWorker(args.upstream, args.output)
    closed = False
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request['command']
                if command == 'reset':
                    result = worker.reset(request['value'])
                elif command == 'step':
                    result = worker.step(request['value'])
                elif command == 'observe':
                    result = worker.observe()
                elif command == 'close':
                    result = worker.close()
                    closed = True
                else:
                    raise ValueError('Unknown command')
                response = dict(ok=True, result=result)
            except Exception as error:
                traceback.print_exc()
                response = dict(ok=False, error=type(error).__name__+': '+str(error))
            reply.write(json.dumps(response, default=json_value, allow_nan=False)+'\n')
            if closed:
                break
    finally:
        if not closed:
            worker.close()


if __name__ == '__main__':
    main()
