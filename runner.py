"""Same as `uninavbench run ...` (kept for backward compatibility)."""
import sys

from uninavbench.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
