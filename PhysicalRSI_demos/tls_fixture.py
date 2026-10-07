"""Temporary test/demo certificates, never a production provisioning service."""

from pathlib import Path
import subprocess


def certificates(root, *, names=("server", "client", "observer", "outsider"), server_ip="127.0.0.1"):
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("TLS fixture certificates require a fresh private directory")
    root.chmod(0o700)

    def openssl(*args):
        result = subprocess.run(["openssl", *args], cwd=root, capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError("Fixture certificate generation failed: " + result.stderr)

    openssl("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-days", "1",
            "-subj", "/CN=physicalRSI simulation fixture CA", "-keyout", "ca.key", "-out", "ca.pem",
            "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    (root / "ca.key").chmod(0o600)
    paths = dict(ca=str(root / "ca.pem"))
    for name in names:
        server = name == "server" or name.endswith("-server")
        openssl("req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
                "-subj", "/CN=physicalRSI fixture " + name, "-keyout", name + ".key", "-out", name + ".csr")
        (root / (name + ".ext")).write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n" +
            ("extendedKeyUsage=serverAuth\nsubjectAltName=IP:" + server_ip + "\n" if server else "extendedKeyUsage=clientAuth\n"))
        openssl("x509", "-req", "-in", name + ".csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
                "-days", "1", "-extfile", name + ".ext", "-out", name + ".pem")
        (root / (name + ".key")).chmod(0o600)
        paths[name] = dict(certificate=str(root / (name + ".pem")), private_key=str(root / (name + ".key")))
    return paths
