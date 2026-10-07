"""Conservative monotonic clock conversion with explicit uncertainty limits.

Four timestamps bound offset without assuming symmetric network delay. Bounds
remain conditional on the declared relative rate error, timestamp error and
continuous clock domains. This is neither a wall-clock synchronizer nor a claim
that an unmeasured physical oscillator satisfies those assumptions.
"""

from dataclasses import asdict, dataclass
import math

from ..contracts import ReconciliationRequired
from ..timing import finite_seconds
from .storage import digest


class ClockUnsynchronized(ReconciliationRequired):
    pass


@dataclass(frozen=True)
class ClockSyncPolicy:
    max_round_trip_seconds: float = .2
    max_offset_width_seconds: float = .05
    max_sample_age_seconds: float = 5
    max_relative_drift_ppm: float = 100
    timestamp_error_seconds: float = .000001

    def __post_init__(self):
        for value in (self.max_round_trip_seconds, self.max_offset_width_seconds, self.max_sample_age_seconds):
            finite_seconds(value)
        finite_seconds(self.timestamp_error_seconds, positive=False)
        finite_seconds(self.max_relative_drift_ppm, positive=False)
        if self.max_relative_drift_ppm >= 1_000_000:
            raise ValueError("Relative clock drift must be less than one second per second")

    @property
    def drift(self):
        return self.max_relative_drift_ppm / 1_000_000


@dataclass(frozen=True)
class ClockMapping:
    policy: ClockSyncPolicy
    local_domain: str
    remote_domain: str
    nonce: str
    local_sent: float
    remote_received: float
    remote_sent: float
    local_received: float

    def __post_init__(self):
        if not isinstance(self.policy, ClockSyncPolicy):
            raise ValueError("Clock conversion requires an explicit policy")
        if any(not isinstance(value, str) or not value for value in (self.local_domain, self.remote_domain, self.nonce)):
            raise ValueError("Clock domains and exchange identity are required")
        stamps = (self.local_sent, self.remote_received, self.remote_sent, self.local_received)
        for value in stamps:
            finite_seconds(value, positive=False)
        if self.local_received < self.local_sent or self.remote_sent < self.remote_received:
            raise ClockUnsynchronized("Clock exchange moved backwards")
        if self.local_received - self.local_sent > self.policy.max_round_trip_seconds:
            raise ClockUnsynchronized("Clock exchange round trip exceeds its declared limit")
        lower, upper = self._bounds(self.local_received, self.local_received)
        if lower > upper:
            raise ClockUnsynchronized("Clock exchange has no feasible offset interval")
        self.offset_bounds(self.local_received, now=self.local_received)

    def _bounds(self, start, end):
        elapsed = self.local_received - self.local_sent
        # Include declared timestamp error and arithmetic rounding in addition
        # to drift during the exchange. Extra width is intentionally retained.
        rounding = 4 * max(math.ulp(value) for value in
                           (self.local_sent, self.remote_received, self.remote_sent, self.local_received, start, end))
        padding = (2 * self.policy.timestamp_error_seconds + rounding + self.policy.drift *
                   (elapsed + 2 * self.policy.timestamp_error_seconds
                    + max(abs(start - self.local_received), abs(end - self.local_received))))
        return self.remote_sent - self.local_received - padding, self.remote_received - self.local_sent + padding

    def offset_bounds(self, start, end=None, *, now):
        end = start if end is None else end
        for value in (start, end, now):
            finite_seconds(value, positive=False)
        if end < start:
            raise ValueError("Clock conversion interval is reversed")
        if now < self.local_received or now - self.local_received > self.policy.max_sample_age_seconds:
            raise ClockUnsynchronized("Clock mapping is expired or the local clock moved backwards")
        lower, upper = self._bounds(start, end)
        if upper - lower > self.policy.max_offset_width_seconds:
            raise ClockUnsynchronized("Clock offset uncertainty exceeds its declared limit")
        return lower, upper

    def remote_deadline(self, local_deadline, *, now):
        if local_deadline <= now:
            raise ClockUnsynchronized("Local deadline has already expired")
        lower, _ = self.offset_bounds(local_deadline, now=now)
        # The device deadline is no later than the local deadline under every
        # offset in the admitted interval. Never use the midpoint here.
        return finite_seconds(local_deadline + lower)

    def local_capture(self, remote_earliest, remote_latest, *, local_sent, local_received):
        for value in (remote_earliest, remote_latest):
            finite_seconds(value, positive=False)
        if remote_latest < remote_earliest:
            raise ClockUnsynchronized("Remote capture interval is reversed")
        lower, upper = self.offset_bounds(local_sent, local_received, now=local_received)
        # The backend separately proves that this sample was freshly captured
        # within this exact observe call. That causal interval can narrow the
        # conversion, but a receive timestamp cannot replace a source sample.
        earliest = max(local_sent, remote_earliest - self.policy.timestamp_error_seconds - upper)
        latest = min(local_received, remote_latest + self.policy.timestamp_error_seconds - lower)
        if earliest > latest:
            raise ClockUnsynchronized("Sensor capture contradicts its clock mapping and causal interval")
        return earliest, latest

    def follows(self, previous):
        if self.local_domain != previous.local_domain or self.remote_domain != previous.remote_domain:
            raise ClockUnsynchronized("Clock domain changed during the session")
        if self.local_sent < previous.local_received:
            raise ClockUnsynchronized("Clock exchange order moved backwards")
        old_lower, old_upper = previous._bounds(self.local_received, self.local_received)
        lower, upper = self._bounds(self.local_received, self.local_received)
        if lower > old_upper or upper < old_lower:
            raise ClockUnsynchronized("New clock exchange contradicts the declared drift bound")

    def evidence(self):
        value = dict(schema="physicalrsi.clock-mapping/v1", **asdict(self))
        return dict(value, sha256=digest(value))
