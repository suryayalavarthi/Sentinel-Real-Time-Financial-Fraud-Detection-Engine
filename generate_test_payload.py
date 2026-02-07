import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"


def main() -> None:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    payload = {"features": {name: 0.0 for name in feature_names}}
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
