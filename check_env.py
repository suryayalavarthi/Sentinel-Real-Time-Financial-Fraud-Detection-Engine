from __future__ import annotations

import importlib.util
import subprocess
import sys


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("\nInstalled packages:")
        print(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        print("\nFailed to list packages.")
        print(exc.stderr.strip())

    pandas_spec = importlib.util.find_spec("pandas")
    if pandas_spec is None:
        print("\nPandas import: NOT FOUND")
    else:
        print("\nPandas import: OK")


if __name__ == "__main__":
    main()
