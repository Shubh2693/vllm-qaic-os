#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Concurrent multi-device dispatcher for run_tests.sh --device-ids.

Adapted from ../../ci_scripts/scheduler.py's DevicePool/Job/Scheduler classes, with
the AOT export-gate/compile-gate machinery removed -- this suite always runs in
eager mode (engine_pool.py sets enforce_eager=True unconditionally), so there is no
AOT export/compile cache to serialise around here.

run_tests.sh hands this a manifest of batches (the same batches --group-by-engine or
--per-test already computed) and a device-ID pool. Each batch's device count is its
TP, parsed from the batch name/id, which always embeds -tp<N>-. Batches run as their
own `pytest --device-id <slice>` subprocess the moment enough devices are free;
--device-id is read by conftest.py's pytest_configure, which sets
QAIC_VISIBLE_DEVICES before collection -- see conftest.py for why that's early enough
(the env var is read lazily by vllm_qaic at LLM(...) construction time, not at
`import vllm` time).

    python3 scheduler.py <manifest> --device-ids 48-63 --output-dir DIR \
        --summary-file summary.txt [--timeout S] [--grace S] [--cooldown S] \
        [--stop-on-fail] [-- <extra pytest args>]

Manifest format (one line per batch, written by run_tests.sh): tab-separated
`<index>\t<name>\t<num_devices>\t<nodeid1>|<nodeid2>|...`.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

_DEFAULT_TIMEOUT_S = 3600.0
_DEFAULT_GRACE_S = 20.0
_DEFAULT_COOLDOWN_S = 10.0
_WAKE_FALLBACK_S = 2.0

# Same tool run_tests.sh's own device_status() already shells out to. Advisory
# only, same convention: if it's missing or unusable (permissions, no devices),
# callers fail open rather than blocking forever.
_QAIC_UTIL = "/opt/qti-aic/tools/qaic-util"
_LIVE_STATUS_TTL_S = 2.0


def _parse_device_pool(spec: str) -> list[int]:
    """"48-63" / "48,50,55" / "48,50-55,60" -> a flat, order-preserving int list."""
    ids: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(chunk))
    if not ids:
        raise ValueError(f"--device-ids parsed to an empty pool: {spec!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"--device-ids contains duplicates: {spec!r}")
    return ids


def _live_device_status(warn_state: dict) -> dict | None:
    """{qid: (status, nsp_free, nsp_total)} from `qaic-util -q`, or None if it
    could not be queried at all (missing binary, no permission, no output).

    This is what DevicePool checks a candidate slice against right before
    handing it to a job -- the in-memory pool only knows what *this scheduler*
    has allocated, not whether a device is actually idle (a leaked process from
    an earlier run, or another user entirely on a shared host, is invisible to
    it otherwise). None means "could not verify" -- callers fail open (trust the
    in-memory pool alone), which is exactly today's behavior without this check.
    """
    if not os.path.exists(_QAIC_UTIL):
        return None
    try:
        # try_acquire() calls this while Scheduler.run() holds its own
        # dispatch-loop lock, so a hung qaic-util stalls every job's
        # release()/dispatch for up to this long -- kept short (rather than
        # this module's other, more generous subprocess timeouts) for that
        # reason; the _LIVE_STATUS_TTL_S cache means it's paid at most once per
        # TTL window regardless.
        result = subprocess.run(
            [_QAIC_UTIL, "-q"], capture_output=True, text=True, timeout=5
        )
        output = result.stdout if result.returncode == 0 else ""
    except Exception:
        output = ""
    if not output.strip():
        if not warn_state.get("warned"):
            print(
                f"[scheduler] WARNING: could not query live device status via "
                f"{_QAIC_UTIL} -q (missing, no permission, or no output) -- "
                "dispatching on the in-memory pool alone, with no live "
                "readiness check, for the rest of this run.",
                file=sys.stderr,
            )
            warn_state["warned"] = True
        return None

    status: dict[int, tuple] = {}
    qid, cur_status, nsp_total, nsp_free = None, "", -1, -1
    for line in output.splitlines():
        stripped = line.strip()
        if re.match(r"^QID\s", stripped):
            match = re.search(r"\d+", stripped)
            qid = int(match.group()) if match else None
            cur_status, nsp_total, nsp_free = "", -1, -1
        elif stripped.startswith("Status:") and qid is not None:
            cur_status = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Nsp Total:") and qid is not None:
            nsp_total = int(stripped.split(":", 1)[1].strip() or -1)
        elif stripped.startswith("Nsp Free:") and qid is not None:
            nsp_free = int(stripped.split(":", 1)[1].strip() or -1)
            status[qid] = (cur_status, nsp_free, nsp_total)
    return status


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:120]


class JobStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class Job:
    job_id: int
    name: str
    nodeids: list
    num_devices: int
    status: str = JobStatus.PENDING
    device_ids: list = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    return_code: int | None = None
    log_path: str | None = None


class DevicePool:
    """In-memory device pool -- this process is the sole authority over what
    *this scheduler* has handed out, so no cross-process locking is needed for
    that bookkeeping (that's a different concern from run_metrics.py's xlsx
    lock, which guards a file every concurrently-running pytest *subprocess*
    writes to). It is NOT the sole authority over the devices themselves,
    though -- another user on a shared host, or a leaked process from an
    earlier run, is invisible to this bookkeeping. try_acquire() cross-checks
    every candidate against a live `qaic-util -q` query before handing it out,
    for exactly that reason.
    """

    def __init__(self, device_ids: list, cooldown_s: float):
        self._free = list(device_ids)
        self._cooldown_s = cooldown_s
        self._cooldown_until: dict[int, float] = {}
        self._lock = threading.Lock()
        self._live_cache: tuple | None = None
        self._warn_state: dict = {}

    def _live_status(self) -> dict | None:
        now = time.monotonic()
        if self._live_cache is not None and now - self._live_cache[0] < _LIVE_STATUS_TTL_S:
            return self._live_cache[1]
        status = _live_device_status(self._warn_state)
        if status is not None:
            self._live_cache = (now, status)
        return status

    def _not_ready(self, device_ids: list) -> list:
        """Which of these IDs are NOT free per a live qaic-util query.

        Empty if the query couldn't be done at all (fail open, same as no check)
        or if every ID checks out; a device qaic-util doesn't even list is left
        alone rather than flagged, for the same fail-open reason.
        """
        live = self._live_status()
        if live is None:
            return []
        not_ready = []
        for d in device_ids:
            entry = live.get(d)
            if entry is None:
                continue
            status, free, total = entry
            if status != "Ready" or free != total:
                not_ready.append(d)
        return not_ready

    def try_acquire(self, count: int) -> list | None:
        """count in-memory-free, live-ready device IDs, or None if fewer than
        count currently qualify.

        Only ever called from Scheduler.run()'s single dispatch thread (never
        concurrently with itself -- release() is what worker threads call), so
        the snapshot-then-verify-then-commit split below doesn't need to guard
        against another try_acquire() racing it for the same candidates.
        """
        if count == 0:
            return []

        with self._lock:
            now = time.monotonic()
            available = [d for d in self._free if self._cooldown_until.get(d, 0) <= now]

        if len(available) < count:
            return None

        # Check every in-memory-free candidate, not just the first `count` of
        # them -- otherwise one externally-busy device occupying an early slot
        # fails the whole acquire even when enough *other* devices in the same
        # pool are genuinely free (this is what actually happened against the
        # leaked-process case that motivated this check in the first place).
        not_ready = self._not_ready(available)
        if not_ready:
            print(
                f"[scheduler] device(s) {not_ready} reported busy by a live "
                "qaic-util query (not by this scheduler's own bookkeeping -- "
                "likely another process on this host); trying the remaining "
                f"genuinely-free device(s) instead",
                file=sys.stderr,
            )
            with self._lock:
                now = time.monotonic()
                for d in not_ready:
                    self._cooldown_until[d] = max(
                        self._cooldown_until.get(d, 0), now + self._cooldown_s
                    )

        ready = [d for d in available if d not in not_ready]
        if len(ready) < count:
            return None

        chosen = ready[:count]
        with self._lock:
            for d in chosen:
                if d in self._free:
                    self._free.remove(d)
        return chosen

    def release(self, device_ids: list) -> None:
        with self._lock:
            now = time.monotonic()
            for d in device_ids:
                self._cooldown_until[d] = now + self._cooldown_s
                self._free.append(d)


class Scheduler:
    def __init__(
        self,
        jobs: list,
        device_ids: list,
        output_dir: Path,
        pytest_extra: list,
        timeout_s: float,
        grace_s: float,
        cooldown_s: float,
        stop_on_fail: bool,
    ):
        self.jobs = jobs
        self.device_pool = DevicePool(device_ids, cooldown_s=cooldown_s)
        self.output_dir = output_dir
        self.pytest_extra = pytest_extra
        self.timeout_s = timeout_s
        self.grace_s = grace_s
        self.stop_on_fail = stop_on_fail

        self.cond = threading.Condition()
        self.print_lock = threading.Lock()
        self.pending: list[Job] = list(jobs)
        self.running: set[int] = set()
        self.abort = False

        oversized = [j for j in jobs if j.num_devices > len(device_ids)]
        if oversized:
            names = ", ".join(f"{j.name} (needs {j.num_devices})" for j in oversized)
            raise ValueError(
                f"pool of {len(device_ids)} device(s) is smaller than {len(oversized)} "
                f"job(s) need: {names}"
            )

    def run(self) -> int:
        threads = []
        while True:
            with self.cond:
                if (not self.pending and not self.running) or (
                    self.abort and not self.running
                ):
                    break

                still_pending = []
                for job in self.pending:
                    if self.abort:
                        still_pending.append(job)
                        continue
                    device_ids = self.device_pool.try_acquire(job.num_devices)
                    if device_ids is None:
                        still_pending.append(job)
                        continue
                    job.device_ids = device_ids
                    job.status = JobStatus.RUNNING
                    self.running.add(job.job_id)
                    print(
                        f"[scheduler] dispatching [{job.job_id + 1}/{len(self.jobs)}] "
                        f"{job.name} (devices {job.device_ids})"
                    )
                    t = threading.Thread(target=self._execute_job, args=(job,), daemon=True)
                    threads.append(t)
                    t.start()
                self.pending = still_pending

                if self.pending or self.running:
                    self.cond.wait(timeout=_WAKE_FALLBACK_S)

        for t in threads:
            t.join()

        lines, failed = self.summary_lines()
        print("\n" + "\n".join(lines))
        return 1 if failed else 0

    def _execute_job(self, job: Job) -> None:
        job.start_time = time.monotonic()
        slug = _slugify(job.name)
        log_path = self.output_dir / f"{job.job_id + 1:03d}_{slug}.log"
        job.log_path = str(log_path)

        cmd = [
            "python",
            "-m",
            "pytest",
            "-s",
            "-p",
            "no:cacheprovider",
            *job.nodeids,
            # Device-free jobs (num_devices == 0, e.g. test_token_math_selftest.py)
            # omit --device-id entirely rather than passing it an empty string --
            # conftest.py's pytest_configure only sets QAIC_VISIBLE_DEVICES when
            # the option has a truthy value either way, but there's no slice to
            # report here at all.
            *(
                ["--device-id", ",".join(str(d) for d in job.device_ids)]
                if job.device_ids
                else []
            ),
            *self.pytest_extra,
        ]

        process = None
        try:
            with open(log_path, "w") as log_file:
                log_file.write(f"### {time.strftime('%c')} :: {job.name}\n")
                log_file.write(f"### {' '.join(cmd)}\n\n")
                log_file.flush()
                # start_new_session=True gives this subprocess its own process
                # group, exactly like run_tests.sh's `set -m` background-job trick --
                # a TERM/KILL to -pgid can only ever hit this job and its children.
                process = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    job.return_code = process.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    self._kill_process_group(process.pid, log_path)
                    job.status = JobStatus.TIMEOUT
                    job.return_code = -1
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.return_code = -1
            print(f"[scheduler] ERROR launching {job.name}: {exc!r}", file=sys.stderr)
        finally:
            job.end_time = time.monotonic()

        if job.status == JobStatus.RUNNING:
            job.status = self._classify(job)

        self._print_job_result(job)

        with self.cond:
            self.device_pool.release(job.device_ids)
            self.running.discard(job.job_id)
            if job.status in (JobStatus.FAIL, JobStatus.TIMEOUT, JobStatus.ERROR):
                if self.stop_on_fail:
                    self.abort = True
            self.cond.notify_all()

    def _kill_process_group(self, pid: int, log_path: Path) -> None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.grace_s
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        with open(log_path, "a") as log_file:
            log_file.write(
                f"\n### killed: exceeded --timeout {self.timeout_s:.0f}s "
                f"(TERM, then KILL after {self.grace_s:.0f}s)\n"
            )

    @staticmethod
    def _classify(job: Job) -> str:
        if job.return_code == 0:
            try:
                tail = Path(job.log_path).read_text(errors="replace")[-4000:]
            except OSError:
                tail = ""
            if "skipped" in tail and "passed" not in tail:
                return JobStatus.SKIP
            return JobStatus.PASS
        return JobStatus.FAIL

    def _print_job_result(self, job: Job) -> None:
        duration = (job.end_time or 0) - (job.start_time or 0)
        with self.print_lock:
            print(
                f"[scheduler] [{job.job_id + 1}/{len(self.jobs)}] {job.name}: "
                f"{job.status} (rc={job.return_code}) in {duration:.1f}s "
                f"(devices {job.device_ids}) -> {job.log_path}"
            )
            if job.status in (JobStatus.FAIL, JobStatus.TIMEOUT, JobStatus.ERROR):
                try:
                    tail = Path(job.log_path).read_text(errors="replace").splitlines()[-15:]
                except OSError:
                    tail = []
                for line in tail:
                    print(f"    {line}")
            sys.stdout.flush()

    def summary_lines(self) -> tuple[list, int]:
        """The summary body (shared by the stdout print and --summary-file) plus
        the failed-job count, which is also this run's exit-code signal."""
        counts = {
            JobStatus.PASS: 0,
            JobStatus.SKIP: 0,
            JobStatus.FAIL: 0,
            JobStatus.TIMEOUT: 0,
            JobStatus.ERROR: 0,
        }
        lines = [f"=== VLM max-context summary (device-pool run): {time.ctime()} ==="]
        for job in self.jobs:
            status = job.status if job.status != JobStatus.PENDING else "NOT RUN"
            counts[status] = counts.get(status, 0) + 1
            duration = (
                (job.end_time or 0) - job.start_time if job.start_time is not None else 0
            )
            lines.append(f"{status:8s} {duration:6.0f}s  {job.name}")
        failed = counts[JobStatus.FAIL] + counts[JobStatus.TIMEOUT] + counts[JobStatus.ERROR]
        lines.append("")
        lines.append(
            f"passed={counts[JobStatus.PASS]} failed={failed} "
            f"skipped={counts[JobStatus.SKIP]} of {len(self.jobs)}"
        )
        if self.abort:
            lines.append("aborted early: --stop-on-fail after a failure")
        return lines, failed


def _load_jobs(manifest_path: Path) -> list:
    jobs = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        index, name, num_devices, nodeids_joined = line.split("\t")
        jobs.append(
            Job(
                job_id=int(index),
                name=name,
                nodeids=nodeids_joined.split("|"),
                num_devices=int(num_devices),
            )
        )
    return jobs


def main() -> int:
    # Split off "-- <extra pytest args>" ourselves rather than declaring a second,
    # argparse.REMAINDER-based positional: REMAINDER after a required positional
    # (manifest) with optional flags in between is a known argparse footgun -- it
    # can swallow those flags into the remainder instead of parsing them, which is
    # exactly what happened here. A plain string split is unambiguous.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        argv, pytest_extra = argv[:sep], argv[sep + 1 :]
    else:
        pytest_extra = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="tab-separated batch manifest from run_tests.sh")
    parser.add_argument("--device-ids", required=True, help='e.g. "48-63" or "48,50-55"')
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_S)
    parser.add_argument("--grace", type=float, default=_DEFAULT_GRACE_S)
    parser.add_argument("--cooldown", type=float, default=_DEFAULT_COOLDOWN_S)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args(argv)

    device_ids = _parse_device_pool(args.device_ids)
    jobs = _load_jobs(Path(args.manifest))
    if not jobs:
        print("[scheduler] no jobs in manifest -- nothing to run")
        return 0

    print(
        f"[scheduler] {len(jobs)} job(s) over a {len(device_ids)}-device pool "
        f"({args.device_ids}); cooldown={args.cooldown:.0f}s grace={args.grace:.0f}s "
        f"timeout={args.timeout:.0f}s"
    )
    scheduler = Scheduler(
        jobs,
        device_ids,
        args.output_dir,
        pytest_extra,
        timeout_s=args.timeout,
        grace_s=args.grace,
        cooldown_s=args.cooldown,
        stop_on_fail=args.stop_on_fail,
    )
    run_start = time.monotonic()
    rc = scheduler.run()
    total_s = time.monotonic() - run_start
    print(f"[scheduler] total wall time: {total_s:.1f}s ({timedelta(seconds=round(total_s))})")

    lines, _failed = scheduler.summary_lines()
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
