from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "benchmarks" / "data" / "fred_macro_context.json"
TEMPLATE_PATH = REPO_ROOT / "benchmarks" / "suite_macro_fred.template.yml"
OUT_PATH = REPO_ROOT / "benchmarks" / "suite_macro_fred.yml"


def indent_block(text: str, indent: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(indent + line for line in lines)


def render_placeholder(template_text: str, placeholder: str, replacement_text: str) -> str:
    pattern = re.compile(rf"(?m)^(?P<indent>[ \t]*){re.escape(placeholder)}$")

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return indent_block(replacement_text, indent)

    rendered_text, count = pattern.subn(repl, template_text)
    if count == 0:
        raise ValueError(f"Template missing placeholder on its own line: {placeholder}")
    return rendered_text


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing context JSON: {DATA_PATH}")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Missing suite template: {TEMPLATE_PATH}")

    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    contexts = payload.get("contexts", {})

    latest_snapshot = contexts.get("latest_snapshot")
    comparison_12m = contexts.get("comparison_12m")

    if not isinstance(latest_snapshot, str) or not isinstance(comparison_12m, str):
        raise ValueError("Context JSON is missing required string fields under 'contexts'.")

    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    rendered_text = render_placeholder(
        template_text,
        "{{ latest_snapshot }}",
        latest_snapshot,
    )
    rendered_text = render_placeholder(
        rendered_text,
        "{{ comparison_12m }}",
        comparison_12m,
    )

    OUT_PATH.write_text(rendered_text, encoding="utf-8")
    print(f"Rendered suite written to: {OUT_PATH}")


if __name__ == "__main__":
    main()