"""Make a phone look RAM-constrained by holding memory in the CROWDio app (debug build).

Needs the debug-only receiver from ``docs/android_ram_limiter`` installed in the app. The app holds
real memory, so the free RAM the phone reports to the Foreman drops, and the planner's memory
check sees it.

  python -m evaluation.experiments.android_ram status
  python -m evaluation.experiments.android_ram hold    --mb 2000
  python -m evaluation.experiments.android_ram target  --free-mb 1200     # leave about 1.2 GB available
  python -m evaluation.experiments.android_ram release
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, Optional

from .android_load import adb, attached

COMPONENT = "com.crowdio.mcc_phase3/.debug.RamBallastReceiver"
ACTION = "com.crowdio.mcc_phase3.RAM_BALLAST"
TOLERANCE_MB = 150


def broadcast(serial: str, mb: int, reserve_mb: int = 512) -> Optional[Dict[str, str]]:
    """Send the ballast request; ``mb=-1`` only reads the state. Returns the receiver's reply fields."""
    r = adb(serial, "shell", "am", "broadcast", "-n", COMPONENT, "-a", ACTION,
            "--ei", "mb", str(mb), "--ei", "reserve_mb", str(reserve_mb))
    match = re.search(r'data="([^"]*)"', r.stdout)
    if not match:
        print(f"[{serial}] no reply (is the debug build with the receiver installed?): {(r.stdout + r.stderr).strip()[:200]}")
        return None
    fields = dict(part.split("=", 1) for part in match.group(1).split() if "=" in part)
    if "held_mb" not in fields:
        print(f"[{serial}] {match.group(1)}")
        return None
    return fields


def show(serial: str, fields: Dict[str, str]) -> None:
    print(f"[{serial}] holding {fields['held_mb']} MiB, {fields['avail_mb']} MiB available to the system"
          + (" (request clamped to keep the safety reserve)" if fields.get("clamped") == "true" else ""))


def target_free(serial: str, free_mb: int, reserve_mb: int) -> Optional[Dict[str, str]]:
    """Adjust the ballast until about ``free_mb`` is available. Android reclaims caches as memory fills,
    so the first guess is usually off and a few corrections are needed."""
    fields = broadcast(serial, -1, reserve_mb)
    for _ in range(5):
        if fields is None:
            return None
        held, avail = int(fields["held_mb"]), int(fields["avail_mb"])
        if abs(avail - free_mb) <= TOLERANCE_MB:
            return fields
        fields = broadcast(serial, max(0, held + (avail - free_mb)), reserve_mb)
    return fields


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["status", "hold", "target", "release"])
    p.add_argument("--serial", action="append", help="adb serial (repeatable); default: every attached phone")
    p.add_argument("--mb", type=int, help="hold: total MiB to hold")
    p.add_argument("--free-mb", type=int, help="target: leave about this many MiB available")
    p.add_argument("--reserve-mb", type=int, default=512, help="MiB the system must always keep available")
    args = p.parse_args(argv)
    if args.action == "hold" and args.mb is None:
        p.error("hold needs --mb")
    if args.action == "target" and args.free_mb is None:
        p.error("target needs --free-mb")

    serials = args.serial or attached()
    if not serials:
        print("no adb device attached")
        return 1
    ok = True
    for serial in serials:
        if args.action == "status":
            fields = broadcast(serial, -1, args.reserve_mb)
        elif args.action == "release":
            fields = broadcast(serial, 0, args.reserve_mb)
        elif args.action == "hold":
            fields = broadcast(serial, args.mb, args.reserve_mb)
        else:
            fields = target_free(serial, args.free_mb, args.reserve_mb)
        if fields is None:
            ok = False
        else:
            show(serial, fields)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
