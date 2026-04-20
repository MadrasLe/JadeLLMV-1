"""
Aggregate public Hugging Face metrics for the Jade model family.

Tracks both official Madras1 models and community quantizations.
Generates separate badges for official downloads and total ecosystem downloads.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_REGISTRY = "models.json"
DEFAULT_BADGES_DIR = "badges"
DEFAULT_SNAPSHOT_PATH = "hf_metrics_snapshot.json"
DEFAULT_HISTORY_PATH = "hf_metrics_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync aggregate Hugging Face metrics for Jade models.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--badges-dir", default=DEFAULT_BADGES_DIR)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--history", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_json(url: str, timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "JadeLLM metrics sync/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_repo_metrics(repo_id: str, timeout: int) -> dict[str, int]:
    encoded_repo = quote(repo_id, safe="/")
    base_url = f"https://huggingface.co/api/models/{encoded_repo}"
    monthly_payload = fetch_json(base_url, timeout)
    all_time_payload = fetch_json(f"{base_url}?expand=downloadsAllTime", timeout)
    return {
        "downloads_30d": int(monthly_payload.get("downloads") or 0),
        "downloads_all_time": int(all_time_payload.get("downloadsAllTime") or 0),
    }


def compact_number(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def make_badge(label: str, message: str, color: str) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
        "namedLogo": "huggingface",
    }


def append_history(history_path: Path, snapshot: dict[str, Any]):
    history = []
    if history_path.exists():
        history = read_json(history_path)
    history.append(snapshot)
    write_json(history_path, history)


def main():
    args = parse_args()
    registry = read_json(Path(args.registry))
    all_models = registry["models"]

    official_models = [m for m in all_models if m.get("official")]
    community_models = [m for m in all_models if m.get("community")]

    print(f"Tracking {len(official_models)} official models and {len(community_models)} community models...")

    def measure_group(models: list[dict]) -> list[dict]:
        results = []
        for model in models:
            print(f"  fetching {model['repo_id']}...")
            try:
                metrics = fetch_repo_metrics(model["repo_id"], args.timeout)
            except Exception as exc:
                print(f"  [warn] failed to fetch {model['repo_id']}: {exc}")
                metrics = {"downloads_30d": 0, "downloads_all_time": 0}
            results.append({
                "slug": model["slug"],
                "display_name": model["display_name"],
                "repo_id": model["repo_id"],
                "hf_url": model["hf_url"],
                "downloads_30d": metrics["downloads_30d"],
                "downloads_all_time": metrics["downloads_all_time"],
            })
        return results

    print("\n[official]")
    measured_official = measure_group(official_models)
    print("\n[community]")
    measured_community = measure_group(community_models)

    official_30d = sum(m["downloads_30d"] for m in measured_official)
    official_all_time = sum(m["downloads_all_time"] for m in measured_official)
    community_30d = sum(m["downloads_30d"] for m in measured_community)
    community_all_time = sum(m["downloads_all_time"] for m in measured_community)
    total_30d = official_30d + community_30d
    total_all_time = official_all_time + community_all_time

    snapshot = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "family": registry["family"]["name"],
        "owner": registry["family"]["owner"],
        "official_model_count": len(official_models),
        "community_model_count": len(community_models),
        "totals": {
            "downloads_30d": total_30d,
            "downloads_all_time": total_all_time,
        },
        "official_totals": {
            "downloads_30d": official_30d,
            "downloads_all_time": official_all_time,
        },
        "community_totals": {
            "downloads_30d": community_30d,
            "downloads_all_time": community_all_time,
        },
        "official_models": measured_official,
        "community_models": measured_community,
    }

    badges_dir = Path(args.badges_dir)

    # Ecosystem-wide badges (official + community)
    write_json(badges_dir / "hf_total_all_time.json", make_badge(
        "HF downloads all-time", compact_number(total_all_time), "brightgreen"
    ))
    write_json(badges_dir / "hf_total_30d.json", make_badge(
        "HF downloads 30d", compact_number(total_30d), "blue"
    ))

    # Official-only badges
    write_json(badges_dir / "hf_official_all_time.json", make_badge(
        "official downloads", compact_number(official_all_time), "green"
    ))

    # Model count badge (official only)
    write_json(badges_dir / "official_models.json", make_badge(
        "official models", str(len(official_models)), "orange"
    ))

    # Community badge
    write_json(badges_dir / "hf_community_all_time.json", make_badge(
        "community downloads", compact_number(community_all_time), "purple"
    ))

    write_json(Path(args.snapshot), snapshot)
    append_history(Path(args.history), snapshot)

    print(f"\n{'='*50}")
    print(f"Official models:    {len(official_models)}")
    print(f"Community models:   {len(community_models)}")
    print(f"Official downloads (all-time): {official_all_time:,}")
    print(f"Community downloads (all-time): {community_all_time:,}")
    print(f"Total ecosystem downloads (all-time): {total_all_time:,}")
    print(f"Total ecosystem downloads (30d): {total_30d:,}")


if __name__ == "__main__":
    main()
