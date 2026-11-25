"""RSSI-aware scheduler: signal strength as a prior, an early warning and a weak-signal penalty."""

import asyncio
import json
import math

import pytest

from foreman.core.scheduling import (
    BandwidthAwareScheduler,
    RssiAwareScheduler,
    RssiCurve,
    calibrate_from_log,
    create_scheduler,
    fit_rssi_curve,
)
from foreman.core.scheduling.scheduler_interface import Task
from foreman.core.worker_metrics import extract_radio_report
from types import SimpleNamespace

MB = 1_000_000.0


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def task(task_id, size=2_000_000):
    return Task(id=task_id, job_id="j", args=json.dumps({"original_args": {}, "data": "x" * size}))


def workers(*names):
    return {n: SimpleNamespace(status="online", success_rate=1.0) for n in names}


def sched(cls=RssiAwareScheduler, clock=None, **options):
    return cls(clock=clock or Clock(), model_size=lambda name: 0, **options)


def measure(s, name, mbytes_per_s):
    s.observe_link_report(name, {"down_bps": mbytes_per_s * MB, "up_bps": mbytes_per_s * MB, "rtt_s": 0.01})


def pick(s, names):
    pool = workers(*names)
    return asyncio.run(s.batch_select_workers([task("t0")], set(names), pool))[0][1]


# ---------------------------------------------------------------- curve and calibration
def test_curve_falls_ten_times_per_decade_of_db():
    curve = RssiCurve(ref_dbm=-50, ref_mbps=60, db_per_decade=20)
    assert curve.mbps(-50) == pytest.approx(60)
    assert curve.mbps(-70) == pytest.approx(6)
    assert curve.ratio(-70, -50) == pytest.approx(0.1)
    assert curve.mbps(-200) == curve.floor_mbps


def test_fit_recovers_the_slope_and_level():
    truth = RssiCurve(ref_dbm=-60, ref_mbps=20, db_per_decade=25)
    pairs = [(r, truth.mbps(r)) for r in range(-80, -40, 2)]
    fitted = fit_rssi_curve(pairs)
    assert fitted.db_per_decade == pytest.approx(25, rel=1e-6)
    assert fitted.mbps(-60) == pytest.approx(20, rel=1e-6)
    assert fitted.samples == len(pairs)


def test_fit_keeps_the_default_with_too_few_or_flat_or_backward_data():
    default = RssiCurve()
    assert fit_rssi_curve([(-50, 30), (-60, 10)]) == default
    flat = fit_rssi_curve([(-56, 15), (-56, 17), (-56.4, 16), (-55.8, 14), (-56, 16)])
    assert flat.db_per_decade == default.db_per_decade and flat.mbps(-56) == pytest.approx(15.6, rel=0.1)
    backward = fit_rssi_curve([(-80 + 5 * i, 50 - 8 * i) for i in range(6)])  # throughput *rises* as signal weakens
    assert backward.db_per_decade == default.db_per_decade


def test_calibrate_from_the_heartbeat_log(tmp_path):
    truth = RssiCurve(ref_dbm=-55, ref_mbps=30, db_per_decade=22)
    log = tmp_path / "heartbeats.jsonl"
    rows = [{"rssi_dbm": r, "bw_down_bps": truth.mbps(r) * 1e6} for r in range(-80, -45, 5)]
    rows += [{"rssi_dbm": -60, "bw_down_bps": None}, {"rssi_dbm": None, "bw_down_bps": 1e6}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    fitted = calibrate_from_log(log)
    assert fitted.samples == 7 and fitted.db_per_decade == pytest.approx(22, rel=1e-6)


# ---------------------------------------------------------------- heartbeat parsing
def test_radio_report_from_an_android_heartbeat():
    assert extract_radio_report({"net": {"rssi": -61, "link_mbps": 433, "bw_down": None}}) == {
        "rssi_dbm": -61.0, "link_mbps": 433.0}
    assert extract_radio_report({"net": {"rssi": 5}}) == {}  # not a plausible dBm value
    assert extract_radio_report({"cpu_usage_percent": 3}) == {}


def test_factory_and_base_hook():
    assert isinstance(create_scheduler("rssi_aware"), RssiAwareScheduler)
    create_scheduler("fifo").observe_radio("w", {"rssi_dbm": -60})  # other schedulers ignore it


# ---------------------------------------------------------------- behaviour
def test_without_radio_reports_it_matches_the_bandwidth_scheduler():
    plain, rssi = sched(BandwidthAwareScheduler), sched()
    for s in (plain, rssi):
        measure(s, "a", 2.0)
        measure(s, "b", 5.0)
    profile = plain.profile(task("t"))
    for w in ("a", "b"):
        est = plain.estimator.estimate(w)
        assert rssi.cost(profile, w, est) == pytest.approx(plain.cost(profile, w, est))
    assert pick(plain, ["a", "b"]) == pick(rssi, ["a", "b"]) == "b"


def test_unmeasured_phones_get_an_rssi_prior_instead_of_the_fleet_median():
    s = sched()
    s.observe_radio("a", {"rssi_dbm": -80})
    s.observe_radio("b", {"rssi_dbm": -48})
    # Both unmeasured: the bandwidth scheduler sees two identical median links and picks "a" by id.
    assert pick(sched(BandwidthAwareScheduler), ["a", "b"]) == "a"
    assert pick(s, ["a", "b"]) == "b"
    est, _, reason = s.adjust("b", s.estimator.estimate("b"))
    assert reason == "rssi prior" and est.down_bps == pytest.approx(s.curve.bytes_per_s(-48))


def test_link_speed_caps_the_prior():
    s = sched()
    s.observe_radio("a", {"rssi_dbm": -40, "link_mbps": 20})
    est, _, _ = s.adjust("a", s.estimator.estimate("a"))
    assert est.down_bps == pytest.approx(20 * s.link_efficiency * MB / 8)


def test_a_signal_drop_since_the_last_measurement_scales_the_estimate_down():
    s = sched()
    for w in ("a", "b"):
        s.observe_radio(w, {"rssi_dbm": -50})
        measure(s, w, 5.0)
    s.observe_radio("a", {"rssi_dbm": -66})
    s.observe_radio("a", {"rssi_dbm": -66})  # smoothed: two reports bring it most of the way
    radio = s._radio["a"]
    assert radio.rssi_at_measurement == -50 and radio.rssi_dbm < -60
    est, _, reason = s.adjust("a", s.estimator.estimate("a"))
    assert "signal fell" in reason
    assert est.down_bps == pytest.approx(5.0 * MB * s.curve.ratio(radio.rssi_dbm, -50))
    # The bandwidth scheduler still sees two equal links and takes "a"; the RSSI one avoids it.
    plain = sched(BandwidthAwareScheduler)
    for w in ("a", "b"):
        measure(plain, w, 5.0)
    assert pick(plain, ["a", "b"]) == "a"
    assert pick(s, ["a", "b"]) == "b"


def test_small_drops_and_better_signal_are_not_acted_on():
    s = sched()
    s.observe_radio("a", {"rssi_dbm": -60})
    measure(s, "a", 5.0)
    s.observe_radio("a", {"rssi_dbm": -64})  # smoothed to -62: under the 4 dB threshold
    assert s.adjust("a", s.estimator.estimate("a"))[2] == "measured"
    s.observe_radio("a", {"rssi_dbm": -40})
    s.observe_radio("a", {"rssi_dbm": -40})
    est, _, _ = s.adjust("a", s.estimator.estimate("a"))
    assert est.down_bps == pytest.approx(5.0 * MB)  # never raised above the measurement


def test_a_new_measurement_resets_the_baseline():
    s = sched()
    s.observe_radio("a", {"rssi_dbm": -50})
    measure(s, "a", 5.0)
    for _ in range(4):
        s.observe_radio("a", {"rssi_dbm": -70})
    assert "signal fell" in s.adjust("a", s.estimator.estimate("a"))[2]
    measure(s, "a", 1.0)  # measured again at the weaker signal: trust the measurement
    assert s.adjust("a", s.estimator.estimate("a"))[2] == "measured"


def test_weak_signal_costs_more():
    s = sched()
    for w, rssi in (("a", -88), ("b", -60)):
        s.observe_radio(w, {"rssi_dbm": rssi})
        s._radio[w].rssi_at_measurement = rssi
        measure(s, w, 5.0)
    profile = s.profile(task("t"))
    ca, cb = (s.cost(profile, w, s.estimator.estimate(w)) for w in ("a", "b"))
    assert ca == pytest.approx(cb * s.weak_penalty)


def test_stale_radio_reports_are_ignored():
    clock = Clock()
    s = sched(clock=clock)
    s.observe_radio("a", {"rssi_dbm": -90})
    clock.t += s.radio_max_age_s + 1
    assert s.adjust("a", s.estimator.estimate("a"))[2] == "no radio"


def test_decisions_and_snapshot_explain_the_choice():
    s = sched()
    s.observe_radio("b", {"rssi_dbm": -48, "link_mbps": 433})
    pick(s, ["b"])
    decision = s.decisions[-1]
    assert decision["rssi_dbm"] == -48 and decision["rssi_reason"] == "rssi prior"
    snap = s.link_snapshot()
    assert snap["radio"]["b"]["link_mbps"] == 433 and snap["rssi_curve"]["samples"] == 0
