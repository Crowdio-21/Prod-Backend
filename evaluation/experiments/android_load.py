"""Generate network load from the phones themselves, as a stand-in for interfering devices.

The PC pushes a small shell loop to each phone over adb. The loop runs on the phone, uses the
phone's own Wi-Fi, and hits the Foreman's probe endpoints (the backend side of the test):

* downlink load: ``GET  /api/probe/<bytes>``  (the Foreman sends random bytes to the phone)
* uplink load:   ``POST /api/probe``          (the phone sends zeros to the Foreman)

Every phone running the loop takes airtime from the others and loads the Foreman's own link,
which is what a nearby interfering device would do. The probe endpoints cap one request at
16 MB, so heavier load means more parallel streams, not bigger requests.

Levels (``--level``):  light = 1 stream, 4 MB, downlink only
                       heavy = 4 streams, 16 MB, both directions

Needs: adb on the PC (USB debugging or ``adb connect``), and ``curl`` on the phone (Android 15
has it). Nothing is installed on the phone; the script lives in /data/local/tmp and ends by
itself after ``--duration`` seconds or when stopped.

  python -m evaluation.experiments.android_load test  --serial RFCY50274FV
  python -m evaluation.experiments.android_load start --level heavy --duration 120
  python -m evaluation.experiments.android_load stop
  python -m evaluation.experiments.android_load start --level light --isolate   # also force-stop the worker app
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

PACKAGE = "com.crowdio.mcc_phase3"
REMOTE_SCRIPT = "/data/local/tmp/crowdio_load.sh"
REMOTE_STOP = "/data/local/tmp/crowdio_load.stop"
LEVELS = {"light": {"streams": 1, "mb": 4, "direction": "down"},
          "heavy": {"streams": 4, "mb": 16, "direction": "both"}}


def find_adb() -> str:
    for candidate in (os.getenv("ADB"), str(Path(os.getenv("LOCALAPPDATA", "")) / "Android/Sdk/platform-tools/adb.exe"),
                      str(Path(os.getenv("ANDROID_HOME", "")) / "platform-tools/adb")):
        if candidate and Path(candidate).exists():
            return candidate
    return "adb"


ADB = find_adb()


def adb(serial: Optional[str], *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    cmd = [ADB] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def attached() -> List[str]:
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True, timeout=30).stdout
    return [line.split()[0] for line in out.splitlines()[1:] if line.strip().endswith("device")]


def loop_script(base: str, streams: int, mb: int, direction: str, duration: int) -> str:
    size = mb * 1024 * 1024
    down = f'curl -s -o /dev/null --max-time 60 "{base}/api/probe/{size}"'
    up = (f'head -c {size} /dev/zero | curl -s -o /dev/null --max-time 60 -X POST '
          f'-H "Content-Type: application/octet-stream" --data-binary @- "{base}/api/probe"')
    steps = {"down": [down], "up": [up], "both": [down, up]}[direction]
    body = "; ".join(steps)
    return f"""#!/system/bin/sh
rm -f {REMOTE_STOP}
END=$(( $(date +%s) + {duration} ))
stream() {{
  while [ "$(date +%s)" -lt "$END" ] && [ ! -f {REMOTE_STOP} ]; do
    {body}
  done
}}
i=0
while [ "$i" -lt {streams} ]; do stream & i=$((i+1)); done
wait
"""


def start(serial: str, base: str, level: str, duration: int, isolate: bool, streams=None, mb=None, direction=None) -> bool:
    cfg = dict(LEVELS[level])
    cfg.update({k: v for k, v in (("streams", streams), ("mb", mb), ("direction", direction)) if v})
    if isolate:
        adb(serial, "shell", "am", "force-stop", PACKAGE)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n") as fh:
        fh.write(loop_script(base, cfg["streams"], cfg["mb"], cfg["direction"], duration))
        local = fh.name
    try:
        pushed = adb(serial, "push", local, REMOTE_SCRIPT)
    finally:
        os.unlink(local)
    if pushed.returncode != 0:
        print(f"[{serial}] push failed: {pushed.stderr.strip()}")
        return False
    result = adb(serial, "shell", f"nohup sh {REMOTE_SCRIPT} >/dev/null 2>&1 &")
    print(f"[{serial}] load started: {cfg['streams']} stream(s), {cfg['mb']} MB, {cfg['direction']}, "
          f"{duration}s{' (worker app stopped)' if isolate else ''}")
    return result.returncode == 0


def stop(serial: str) -> None:
    adb(serial, "shell", f"touch {REMOTE_STOP}")  # the loop checks this between requests
    # "[c]rowdio_load" cannot match the shell that runs this command, which a plain pattern would.
    adb(serial, "shell", "pkill -f '[c]rowdio_load.sh'; pkill -x curl")
    print(f"[{serial}] load stopped")


def self_test(serial: str, base: str) -> bool:
    """One 16 MB download and one upload from the phone, reporting its own measured speed."""
    size = 16 * 1024 * 1024
    ok = True
    for name, cmd in (
        ("down", f'curl -s -o /dev/null -w "%{{speed_download}} %{{size_download}} %{{http_code}}" "{base}/api/probe/{size}"'),
        ("up", f'head -c {size} /dev/zero | curl -s -o /dev/null -w "%{{speed_upload}} %{{size_upload}} %{{http_code}}" '
               f'-X POST -H "Content-Type: application/octet-stream" --data-binary @- "{base}/api/probe"'),
    ):
        r = adb(serial, "shell", cmd, timeout=120)
        try:
            speed, nbytes, code = r.stdout.split()[-3:]
            mbit = float(speed) * 8 / 1e6
            print(f"[{serial}] {name}: {mbit:.1f} Mbit/s, {int(float(nbytes)) / 1e6:.1f} MB, HTTP {code}")
            ok &= code == "200"
        except (ValueError, IndexError):
            print(f"[{serial}] {name}: failed ({r.stdout.strip() or r.stderr.strip()})")
            ok = False
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["start", "stop", "test"])
    p.add_argument("--serial", action="append", help="adb serial (repeatable); default: every attached phone")
    p.add_argument("--foreman", default=os.getenv("CROWDIO_PUBLIC_URL", "http://192.168.1.238:8000"),
                   help="address the PHONE uses to reach the Foreman REST port (not localhost)")
    p.add_argument("--level", choices=sorted(LEVELS), default="light")
    p.add_argument("--streams", type=int)
    p.add_argument("--mb", type=int, choices=range(1, 17), metavar="1-16")
    p.add_argument("--direction", choices=["down", "up", "both"])
    p.add_argument("--duration", type=int, default=120, help="seconds before the phone stops by itself")
    p.add_argument("--isolate", action="store_true",
                   help="force-stop the CROWDio worker app on that phone, so it is only an interferer")
    args = p.parse_args(argv)

    serials = args.serial or attached()
    if not serials:
        print("no adb device attached (check USB debugging, or run: adb connect <phone ip>:5555)")
        return 1
    ok = True
    for serial in serials:
        if args.action == "start":
            ok &= start(serial, args.foreman, args.level, args.duration, args.isolate,
                        args.streams, args.mb, args.direction)
        elif args.action == "stop":
            stop(serial)
        else:
            ok &= self_test(serial, args.foreman)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
