#!/usr/bin/env python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.joint_clip import main


if __name__ == "__main__":
    main()
