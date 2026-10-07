"""Identify the active OpenGL context before attributing native work to a GPU."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest


def current_renderer_identity(*, require_nvidia=False):
    from OpenGL import GL

    identity = {}
    for name, key in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                      ("version", GL.GL_VERSION)):
        value = GL.glGetString(key)
        if not isinstance(value, bytes) or not value:
            raise RuntimeError("No readable current OpenGL context")
        identity[name] = value.decode("utf-8", errors="strict")
    if require_nvidia and identity["vendor"] != "NVIDIA Corporation":
        raise RuntimeError("NVIDIA renderer required; observed " + identity["vendor"])
    return {**identity, "implementation_sha256": file_digest(Path(__file__))}
