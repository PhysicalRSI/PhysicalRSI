from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from PhysicalRSI_core.infra.language import ModelConfig
from PhysicalRSI_core.infra.proposal_model import ModelProposal
from PhysicalRSI_core.infra.storage import canonical, read_json
from PhysicalRSI_core.self_harness.proposals import ProposalLimits
from test_structured_proposals import setup


@contextmanager
def provider(mode="normal"):
    requests, disconnected = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(dict(path=self.path, body=body, authorization=self.headers.get("Authorization")))
            history = body.get("input", body.get("messages"))
            request = json.loads(history[-1]["content"])
            entries = request.get("editable", {})
            candidates = []
            if entries and entries["control.json"]["value"] != dict(increment=1):
                candidates = [dict(rationale="Fixture response: exercise the proposal protocol", edits=[dict(
                    path="control.json", before_sha256=entries["control.json"]["sha256"], value=dict(increment=1))])]
            content = canonical(dict(parent_sha256=request.get("parent_sha256"), candidates=candidates)).decode()
            if mode == "echo":
                content = "fixture-credential-never-save-this"
            if self.path.endswith("/responses"):
                response = dict(id="fixture-response", model="fixture-model", status="completed",
                                output=[dict(type="message", content=[dict(type="output_text", text=content)])],
                                usage=dict(input_tokens=10, output_tokens=20, total_tokens=30))
                if mode == "incomplete":
                    response["status"] = "incomplete"
            else:
                response = dict(id="fixture-response", model="fixture-model", choices=[dict(
                    finish_reason="stop", message=dict(role="assistant", content=content))],
                    usage=dict(prompt_tokens=10, completion_tokens=20, total_tokens=30))
                if mode == "tools":
                    response["choices"][0] = dict(finish_reason="tool_calls", message=dict(
                        role="assistant", content=content, tool_calls=[dict(id="not-executed", function=dict(
                            name="execute", arguments='{"command":"must not run"}'))]))
            payload = canonical(response)
            if mode == "oversized":
                payload = b"x" * 100000
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                if mode == "trickle":
                    for value in payload:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(.05)
                else:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests, disconnected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def model(endpoint, protocol="chat_completions"):
    return ModelProposal(ModelConfig(model="fixture-model", base_url=endpoint,
                                     api_key_env="PHYSICALRSI_TEST_PROPOSAL_KEY", protocol=protocol),
                         provider_revision="fixture-only-no-model-inference/1")


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("PHYSICALRSI_TEST_PROPOSAL_KEY", "fixture-credential-never-save-this")


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_http_model_adapter_drives_the_shared_loop_and_preserves_reported_usage(tmp_path, protocol):
    with provider() as (endpoint, requests, _):
        parent, _, suite, campaign = setup(tmp_path, model(endpoint, protocol))
        result = campaign.run(parent)
        assert [r["outcome"] for r in result["rounds"]] == ["inherited", "retained"]
        assert len(requests) == 2 and len(suite.resets) == 6
        assert all(row["authorization"] == "Bearer fixture-credential-never-save-this" for row in requests)
        assert all(row["body"]["tools"] == [] for row in requests)
        call = read_json(tmp_path / "campaign/rounds/round-0001/proposal/call.json")
        assert call["reply"]["provider"]["response_id"] == "fixture-response"
        assert call["reply"]["usage"]["total_tokens"] == 30
        assert result["current"]["harness"]["costs"]["provider_reported_usage"]["total_tokens"] == 30
        assert result["qualification"] is None
        assert campaign.run(parent) == result
        assert len(requests) == 2
    assert all("fixture-credential-never-save-this" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_total_model_deadline_stops_trickling_response_and_owned_worker():
    with provider("trickle") as (endpoint, requests, disconnected):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="total deadline"):
            model(endpoint).generate({}, ProposalLimits(seconds=3))
        assert 3 <= time.monotonic() - started < 6
        assert len(requests) == 1
        assert disconnected.wait(2)


@pytest.mark.parametrize("mode", ["redirect", "oversized", "echo"])
def test_model_boundary_rejects_redirects_oversize_and_credential_echo(mode):
    with provider(mode) as (endpoint, requests, _):
        with pytest.raises(RuntimeError, match="transport failed"):
            model(endpoint).generate({}, ProposalLimits(seconds=10, response_bytes=2000))
        assert len(requests) == 1


@pytest.mark.parametrize("mode,protocol", [("tools", "chat_completions"), ("incomplete", "responses")])
def test_tools_or_incomplete_responses_cannot_produce_a_candidate(tmp_path, mode, protocol):
    with provider(mode) as (endpoint, requests, _):
        parent, _, suite, campaign = setup(tmp_path, model(endpoint, protocol))
        result = campaign.run(parent)
        assert result["current"]["harness"] == parent
        assert all(row["outcome"] == "retained" for row in result["rounds"])
        assert len(requests) == 2 and suite.resets == [3, 3]
        report = read_json(tmp_path / "campaign/rounds/round-0001/proposal/materialization.json")
        assert report["result"]["rejected"] == [dict(index=None, reason="invalid_response")]
