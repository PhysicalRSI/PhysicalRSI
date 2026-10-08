"""Trusted deployment worker: isolated Codex emits data for StructuredProposer."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tomllib

from PhysicalRSI_core.infra.proposal_model import INSTRUCTION
from PhysicalRSI_core.infra.storage import atomic_json, canonical, file_digest, strict_json
from PhysicalRSI_baselines.robotworld.isolation import minimal_proc_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('deployment', 'input', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    deployment = strict_json(args.deployment.read_text())
    payload = strict_json(args.input.read_text())
    limits, request = payload['limits'], payload['request']
    if len(canonical(request)) > limits['request_bytes']:
        raise ValueError('Request too large')
    if request.get('schema') != 'physicalrsi.json-proposal-request/v1':
        raise ValueError('Only core structured proposal requests are accepted')
    sources = deployment['sources']
    if not sources:
        raise ValueError('Pin deployment source inputs')

    def stable():
        for name, expected in sources.items():
            if file_digest(Path(name)) != expected:
                raise ValueError('Deployment source changed')

    stable()
    upstream = Path(deployment['upstream']).resolve(strict=True)
    expected_sources = {str(p.resolve()) for p in (upstream / 'environment').rglob('*.py')}
    if not expected_sources <= set(sources):
        raise ValueError('Deployment source closure omits upstream Python inputs')
    home = Path(deployment['codex_home'])
    config = tomllib.loads((home / 'config.toml').read_text())
    if config.get('model_provider', 'openai') != deployment['provider']:
        raise ValueError('Provider differs from the frozen deployment')
    sys.path.insert(0, str(upstream))
    from environment.runtime import isolated_codex
    from environment.runtime.codex_session import CodexSession

    os.environ['PATH'] = str(Path(deployment['bubblewrap']).parent) + os.pathsep + os.environ['PATH']
    os.environ['WORLD_CODEX_DISABLE_CODE_MODE'] = '1'
    os.environ['TMPDIR'] = deployment['runtime_tmp']
    tempfile.tempdir = deployment['runtime_tmp']
    original = isolated_codex.sandbox_command

    def sandbox(*positional, **kwargs):
        command = minimal_proc_command(original(*positional, **kwargs))
        settings = []
        for name in ('http_proxy', 'https_proxy', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'):
            if name in os.environ:
                settings += ['--setenv', name, os.environ[name]]
        index = command.index('--clearenv') + 1
        command[index:index] = settings
        return command

    isolated_codex.sandbox_command = sandbox
    output = args.output.parent / 'isolated-agent'
    output.mkdir(exist_ok=False)
    text, usage, calls = '', None, []
    deadline = time.monotonic() + limits['seconds']
    with isolated_codex.IsolatedCodex(deployment['build_manifest'], output, home) as relay:
        os.environ['WORLD_CODEX_SOCKET'] = str(relay.socket_path)
        with CodexSession(deployment['build_manifest'], output / 'client') as session:
            thread = session.rpc('thread/start', dict(
                cwd='/workspace', approvalPolicy='never', sandbox='danger-full-access',
                ephemeral=True, model=deployment['model'], baseInstructions=INSTRUCTION,
                config={'model_reasoning_effort': deployment['reasoning_effort'],
                        'features.shell_tool': False, 'features.apps': False,
                        'features.plugins': False, 'features.multi_agent': False,
                        'web_search': 'disabled'}))['thread']['id']
            turn = session.rpc('turn/start', {'threadId': thread, 'input': [
                {'type': 'text', 'text': canonical(request).decode(), 'text_elements': []}]})['turn']['id']
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Proposal inference deadline reached')
                message = session.receive(remaining)
                method, params = message.get('method'), message.get('params', {})
                if method == 'item/agentMessage/delta':
                    if params.get('threadId') != thread or params.get('turnId') != turn:
                        raise ValueError('Wrong proposal event binding')
                    text += params['delta']
                    if len(text.encode()) > limits['response_bytes']:
                        raise ValueError('Proposal text exceeds byte allowance')
                elif method == 'thread/tokenUsage/updated':
                    if params.get('threadId') == thread:
                        usage = params.get('tokenUsage')
                elif method == 'turn/completed':
                    if (params.get('threadId') != thread or params['turn']['id'] != turn
                            or params['turn']['status'] != 'completed'):
                        raise RuntimeError('Proposal turn did not complete normally')
                    break
                elif method and 'id' in message:
                    calls.append({'method': method})
                    session.send({'id': message['id'], 'error': {
                        'code': -32601, 'message': 'System 2 proposals are tool-free JSON data'}})
    stable()
    reply = dict(text=text, usage=usage, tool_calls=calls,
                 provider={'model': deployment['model'], 'provider': deployment['provider'],
                           'status': 'completed', 'identity_role': 'requested model; weights provider-owned'})
    if len(canonical(reply)) + 1 > limits['response_bytes']:
        raise ValueError('Proposal envelope exceeds byte allowance')
    atomic_json(args.output, reply)


if __name__ == '__main__':
    main()
