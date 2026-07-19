"""v4 fine-tune: identical QLoRA configuration to v3, only the training pairs differ.

The whole point of the v-series is that everything except the data is held fixed, so this is a
thin wrapper over finetune_qg_v3 with the v4 paths patched in.

    python src/question_gen/finetune_qg_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "question_gen"))

import finetune_qg_v3 as base  # noqa: E402

base.PAIRS = REPO / "data" / "interim" / "qg_v4_pairs.json"
base.ADAPTER_DIR = REPO / "models" / "qg_finetune_qwen3b_v4"

if __name__ == "__main__":
    raise SystemExit(base.main())
