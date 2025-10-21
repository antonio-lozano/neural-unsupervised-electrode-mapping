from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = SCRIPTS_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rf_demo.data_loader import extract_demo_bundle  # type: ignore[import]
from rf_demo.config import MONKEY_CHOICES, canonical_alias, FREQ_BANDS, resolve_monkey_alias  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a lightweight demo dataset from the full repository")
    parser.add_argument("--monkey", default=MONKEY_CHOICES[-1], choices=list(MONKEY_CHOICES), help="Subject to sample (use 'monkey_L' or 'monkey_A')")
    parser.add_argument("--freq-band", help="Frequency band to extract (if not specified, extracts all bands)")
    parser.add_argument("--window", default="15s", choices=["15s", "full"], help="Data window size")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if the target archive already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monkey_alias = canonical_alias(args.monkey)
    monkey_actual = resolve_monkey_alias(monkey_alias)
    # Ensure only monkey_L and monkey_A are used
    if monkey_alias not in ("monkey_L", "monkey_A"):
        raise ValueError("Only 'monkey_L' and 'monkey_A' are valid monkey aliases for demo extraction.")
    
    if args.freq_band:
        bands_to_extract = [args.freq_band]
    else:
        bands_to_extract = FREQ_BANDS
    
    for freq_band in bands_to_extract:
        try:
            path = extract_demo_bundle(
                monkey=monkey_actual,
                freq_band=freq_band,
                window=args.window,
                overwrite=not args.no_overwrite,
            )
            print(f"Saved demo bundle to {path}")
        except Exception as e:
            print(f"Failed to extract {freq_band}: {e}")


if __name__ == "__main__":
    main()
