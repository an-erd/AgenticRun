from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agenticrun.core.models import RunState


class OutputAgent:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_batch(self, states: list[RunState]) -> None:
        rows = [s.as_flat_dict() for s in states if s.run_record]
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(self.out_dir / "runs_normalized.csv", index=False)
        df.to_excel(self.out_dir / "runs_normalized.xlsx", index=False)
        with open(self.out_dir / "runs_normalized.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        markdown_lines = ["# AgenticRun batch summary", ""]
        for state in states:
            if not state.run_record:
                continue
            markdown_lines.append(f"## {state.run_record.run_date} – {state.run_record.title}")
            markdown_lines.append(state.llm_summary or state.analysis.summary)
            markdown_lines.append("")
        (self.out_dir / "summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    def append_batch(self, states: list[RunState]) -> None:
        """Append normalized exports (used by chunked bulk-import between checkpoints)."""
        rows = [s.as_flat_dict() for s in states if s.run_record]
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        csv_path = self.out_dir / "runs_normalized.csv"
        if csv_path.is_file():
            df_old = pd.read_csv(csv_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(csv_path, index=False)

        xlsx_path = self.out_dir / "runs_normalized.xlsx"
        if xlsx_path.is_file():
            df_old_x = pd.read_excel(xlsx_path)
            df_x = pd.concat([df_old_x, df_new], ignore_index=True)
        else:
            df_x = df
        df_x.to_excel(xlsx_path, index=False)

        json_path = self.out_dir / "runs_normalized.json"
        if json_path.is_file():
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            combined = existing + rows
        else:
            combined = rows
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        section_lines: list[str] = []
        for state in states:
            if not state.run_record:
                continue
            section_lines.append(f"## {state.run_record.run_date} – {state.run_record.title}")
            section_lines.append(state.llm_summary or state.analysis.summary)
            section_lines.append("")
        summary_path = self.out_dir / "summary.md"
        body = "\n".join(section_lines)
        if summary_path.is_file():
            prev = summary_path.read_text(encoding="utf-8")
            if prev and not prev.endswith("\n"):
                prev += "\n"
            summary_path.write_text(prev + "\n" + body, encoding="utf-8")
        else:
            markdown_lines = ["# AgenticRun batch summary", ""] + section_lines
            summary_path.write_text("\n".join(markdown_lines), encoding="utf-8")
