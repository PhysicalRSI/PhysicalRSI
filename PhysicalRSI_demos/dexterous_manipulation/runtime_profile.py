"""Shared Codex session setup; concrete environments supply skills and tools."""
import os
from pathlib import Path
from urllib.parse import urlparse
from hybrid_rollout.robodojo.settings import DEFAULT_CODEX_IMAGE_MAX_EDGE
from hybrid_rollout.robodojo.skill.image_preview import image_max_edge

from .common import file_hash


class RuntimeProfile:
    provider = "physicalrsi_astra"
    allow_same_thread_network_continue = False

    def __init__(self, *, base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY", memory=None,
                 auth="api", login_home=None):
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query:
            raise ValueError("Use an explicit HTTPS Responses endpoint without embedded credentials")
        self.base_url, self.key_env = base_url, key_env
        self.memory = Path(memory).resolve() if memory else None
        if auth not in {"api", "chatgpt"}:
            raise ValueError("Unknown auth mode")
        self.auth = auth
        self.login_home = Path(login_home or os.environ.get("CODEX_HOME", str(Path.home()/".codex"))).resolve()
        if auth == "chatgpt":
            self.provider = "openai"

    def identity(self):
        return dict(provider=self.provider, base_url=self.base_url, key_env=self.key_env, auth=self.auth,
            model="gpt-6-astra", effort="xhigh", skill_sha256=file_hash(self.skill_root/"SKILL.md"),
            image_max_edge=image_max_edge(os.environ.get("CODEX_IMAGE_MAX_EDGE",str(DEFAULT_CODEX_IMAGE_MAX_EDGE))),
            memory_sha256=file_hash(self.memory) if self.memory else None)

    def tool_specs(self):
        raise NotImplementedError("Environment adapter must supply its tool schemas")

    def prepare_workspace(self, audit):
        raise NotImplementedError("Environment adapter must prepare its skill workspace")

    def codex_config(self, audit, agent):
        config = {"model_provider": self.provider,
            "features.apps": False,
            "sqlite_home": str(Path(audit)/"runtime_db"), "log_dir": str(Path(audit)/"runtime_logs"),
            "shell_environment_policy.inherit": "core",
            "shell_environment_policy.exclude": ["*KEY*", "*TOKEN*", "*SECRET*", "ROS_*", "RMW_*"],
            "permissions.rollout_agent.filesystem": {str(Path(audit).parent):"read", str(agent):"write"}}
        if self.auth == "api":
            config[f"model_providers.{self.provider}"] = dict(name="PhysicalRSI Astra", base_url=self.base_url,
                env_key=self.key_env, wire_api="responses")
        else:
            config.update(cli_auth_credentials_store="file", forced_login_method="chatgpt")
        return config

    def child_environment(self, workspace):
        # No inherited developer Codex config, hooks, MCPs, or auth files.
        private = Path(workspace)/"private_codex_home"
        private.mkdir(mode=0o700)
        env = {k:v for k,v in os.environ.items()
               if k in {"PATH", "HOME", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", self.key_env,
                        "http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                        "ALL_PROXY", "all_proxy"}}
        if self.auth == "api" and not env.get(self.key_env):
            raise ValueError(f"Set {self.key_env} in the runner environment; credentials are never written to config")
        if self.auth == "chatgpt":
            source = self.login_home / "auth.json"
            if not source.is_file():
                raise ValueError("Run codex login in the supplied login home first")
            # Reuse the existing authorized CLI login without printing credentials
            # or altering its configuration. No auth data enters evidence JSON.
            (private / "auth.json").symlink_to(source)
        env["CODEX_HOME"] = str(private)
        return env

    def cleanup_workspace(self, workspace):
        """Detach temporary CLI links only after the app-server has stopped.

        Core campaign evidence must contain physical files. The original login
        and executable targets are never changed or included in the evidence.
        """
        private = Path(workspace)/"private_codex_home"
        if private.is_dir():
            # A CLI refresh can replace its temporary symlink with a local
            # credential file. Remove that private copy too, never its target.
            (private/"auth.json").unlink(missing_ok=True)
            for path in private.rglob("*"):
                if path.is_symlink():
                    path.unlink()

    def initial_prompt(self, rollout):
        raise NotImplementedError("Environment adapter must supply its task prompt")
