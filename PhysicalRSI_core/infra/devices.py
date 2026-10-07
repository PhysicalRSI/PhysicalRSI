"""Durable exclusive device ownership for cooperating local POSIX processes.

All clients of a device must share this registry on one controller host. SQLite
transactions acquire whole resource sets; flock proves that an owner is still
inside its lease. A crash releases the OS lock but leaves the durable claim.
Claims never expire: only normal completion or recorded reconciliation frees a
device. This is neither a distributed fencing service nor a hardware interlock.
"""

import fcntl
import json
import sqlite3
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import ReconciliationRequired
from ..embodiment import resource_names
from .storage import canonical, digest, identifier


class DeviceBusy(RuntimeError):
    """A live experiment owns at least one requested device."""


def _now():
    return datetime.now(timezone.utc).isoformat()


class DeviceRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS leases (token TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS resources ("
                "name TEXT PRIMARY KEY, token TEXT NOT NULL REFERENCES leases(token))"
            )

    def identity(self):
        return {"schema": "physicalrsi.device-registry/v1", "root": str(self.root)}

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.root / "devices.sqlite3", timeout=5)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _lock(self, token):
        stream = (self.root / (identifier(token) + ".lock")).open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return None
        except BaseException:
            stream.close()
            raise
        return stream

    def _active(self, token):
        stream = self._lock(token)
        if stream is None:
            return True
        stream.close()
        return False

    @staticmethod
    def _read(connection, token):
        row = connection.execute("SELECT record FROM leases WHERE token = ?", (token,)).fetchone()
        if row is None:
            raise KeyError("Unknown device lease: " + token)
        return json.loads(row[0])

    @staticmethod
    def _write(connection, record):
        connection.execute("UPDATE leases SET record = ? WHERE token = ?",
                           (canonical(record).decode(), record["token"]))

    def inspect(self, token):
        """Report persisted state and actual local owner liveness separately."""
        identifier(token)
        with self._transaction() as connection:
            record = self._read(connection, token)
            record["owner_active"] = self._active(token)
            return record

    def occupied(self):
        with self._transaction() as connection:
            return dict(connection.execute("SELECT name, token FROM resources ORDER BY name"))

    @contextmanager
    def lease(self, resources, *, owner):
        resources = resource_names(resources)
        if not resources or not isinstance(owner, str) or not owner.strip():
            raise ValueError("A device lease requires resources and an owner identity")
        token = uuid.uuid4().hex
        stream = None
        try:
            with self._transaction() as connection:
                # Check the entire set before changing any ownership.
                for name in resources:
                    row = connection.execute(
                        "SELECT token FROM resources WHERE name = ?", (name,)
                    ).fetchone()
                    if row is not None:
                        message = f"Device {name!r} belongs to lease {row[0]}"
                        if self._active(row[0]):
                            raise DeviceBusy(message)
                        raise ReconciliationRequired(message + "; inspect before reuse")
                # Create the owner lock only after admission so ordinary busy
                # submissions do not leave an unbounded set of unused files.
                stream = self._lock(token)
                if stream is None:
                    raise DeviceBusy("Device lease token already in use")
                record = dict(schema="physicalrsi.device-lease/v1", token=token,
                              owner=owner, resources=list(resources), state="held",
                              acquired_at=_now(), registry=self.identity())
                connection.execute("INSERT INTO leases VALUES (?, ?)",
                                   (token, canonical(record).decode()))
                connection.executemany("INSERT INTO resources VALUES (?, ?)",
                                       [(name, token) for name in resources])
            try:
                yield deepcopy(record)
            except BaseException as error:
                # An error cannot prove that a remote action or reset stopped.
                with self._transaction() as connection:
                    record.update(state="needs_reconciliation", finished_at=_now(),
                                  error=type(error).__name__ + ": " + str(error))
                    self._write(connection, record)
                raise
            else:
                with self._transaction() as connection:
                    record.update(state="released", finished_at=_now())
                    self._write(connection, record)
                    connection.execute("DELETE FROM resources WHERE token = ?", (token,))
        finally:
            if stream is not None:
                stream.close()

    def reconcile(self, token, *, operator, reason, evidence):
        """Attest that all devices in an inactive lease are ready for reuse.

        The caller must independently establish device quiescence. This records
        that decision; it cannot stop a device, retry a trial, or declare success.
        A live lease cannot be cleared, even by its owner or after a timeout.
        """
        identifier(token)
        if (not isinstance(operator, str) or not operator.strip()
                or not isinstance(reason, str) or not reason.strip()
                or not isinstance(evidence, dict) or not evidence):
            raise ValueError("Reconciliation requires an operator, reason and evidence")
        resolution = json.loads(canonical(dict(operator=operator, reason=reason,
                                              evidence=evidence, evidence_sha256=digest(evidence))))
        with self._transaction() as connection:
            record = self._read(connection, token)
            stream = self._lock(token)
            if stream is None:
                raise DeviceBusy("Cannot reconcile a live device owner: " + token)
            try:
                if record["state"] == "reconciled":
                    if record["reconciliation"] != resolution:
                        raise ValueError("Device reconciliation is already recorded")
                    return record
                if record["state"] not in {"held", "needs_reconciliation"}:
                    raise ValueError("Only uncertain device leases can be reconciled")
                record.update(state="reconciled", reconciled_at=_now(), reconciliation=resolution)
                self._write(connection, record)
                connection.execute("DELETE FROM resources WHERE token = ?", (token,))
                return record
            finally:
                stream.close()
