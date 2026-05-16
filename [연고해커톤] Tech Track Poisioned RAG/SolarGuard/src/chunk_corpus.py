"""
chunk_corpus.py — Build retrieval chunks from parsed pages.

Input:
    parsed_corpus/pdfplumber/pages.jsonl

Outputs:
    parsed_corpus/pdfplumber/chunks_slide.jsonl
    parsed_corpus/pdfplumber/chunks_block.jsonl

Strategies:
    - chunk_slide: fixed character windows.
    - chunk_block: split text into blocks, then pack blocks into chunks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .logging_utils import log_block  # type: ignore
    from .parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore
except ImportError:
    from logging_utils import log_block  # type: ignore
    from parse_corpus import DEFAULT_CONFIG_PATH, load_config  # type: ignore


@dataclass
class ChunkingConfig:
    active_strategy: str = "chunk_block"
    strategies: tuple[str, ...] = ("chunk_slide", "chunk_block")
    max_chunk_chars: int = 1200
    overlap_chars: int = 400
    block_min_chars: int = 1200
    block_max_chars: int = 1600
    slide_output_filename: str = "chunks_slide.jsonl"
    block_output_filename: str = "chunks_block.jsonl"


STRATEGY_OUTPUTS = {
    "chunk_slide": "slide_output_filename",
    "chunk_block": "block_output_filename",
}


EMAIL_BOUNDARY_RE = re.compile(
    r"^\s*(from|to|cc|bcc|subject|date|sent):\s+",
    re.IGNORECASE,
)
MESSAGE_SEPARATOR_RE = re.compile(
    r"^\s*(-{2,}\s*(original message|forwarded by|forwarded message)?\s*-{2,}|_{3,})\s*$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    normalized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def table_rows_to_text(rows: list[list[Any]]) -> str:
    rendered_rows = []
    for row in rows:
        cells = [str(cell or "").strip() for cell in row]
        cells = [cell for cell in cells if cell]
        if cells:
            rendered_rows.append(" | ".join(cells))
    return "\n".join(rendered_rows)


def record_text(record: dict[str, Any]) -> str:
    parts = []
    page_text = normalize_text(record.get("text", ""))
    if page_text:
        parts.append(page_text)

    for table in record.get("tables", []):
        if not isinstance(table, dict):
            continue
        table_text = normalize_text(table.get("text") or table_rows_to_text(table.get("rows", [])))
        if not table_text:
            continue
        table_index = table.get("table_index", "")
        label = f"[Table {table_index}]" if table_index else "[Table]"
        parts.append(f"{label}\n{table_text}")

    return normalize_text("\n\n".join(parts))


def split_fixed_windows(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    text = normalize_text(text)
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("chunking.max_chunk_chars must be greater than 0")
    if overlap_chars < 0:
        raise ValueError("chunking.overlap_chars must be greater than or equal to 0")
    if overlap_chars >= max_chars:
        raise ValueError("chunking.overlap_chars must be smaller than chunking.max_chunk_chars")

    chunks: list[tuple[str, int, int]] = []
    start = 0
    step = max_chars - overlap_chars
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))
        if end >= text_length:
            break
        start += step

    return chunks


def strategy_output_filename(config: ChunkingConfig, strategy: str) -> str:
    attr = STRATEGY_OUTPUTS.get(strategy)
    if not attr:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
    return str(getattr(config, attr))


def make_chunk(
    *,
    strategy: str,
    source: str,
    page: int,
    backend: str,
    sequence: int,
    text: str,
    start_char: int,
    end_char: int,
) -> dict[str, Any]:
    return {
        "chunk_id": f"{Path(source).stem}:p{page}:{strategy}:{sequence:04d}",
        "source": source,
        "page": page,
        "backend": backend,
        "kind": strategy,
        "sequence": sequence,
        "section": f"page {page} {strategy} {sequence}",
        "text": text,
        "start_char": start_char,
        "end_char": end_char,
    }


def build_slide_chunks_for_record(record: dict[str, Any], config: ChunkingConfig) -> list[dict[str, Any]]:
    source = str(record.get("source", ""))
    page = int(record.get("page") or 0)
    backend = str(record.get("backend", ""))
    text = record_text(record)

    chunks = []
    for sequence, (chunk_text, start_char, end_char) in enumerate(
        split_fixed_windows(text, config.max_chunk_chars, config.overlap_chars),
        start=1,
    ):
        chunks.append(
            make_chunk(
                strategy="chunk_slide",
                source=source,
                page=page,
                backend=backend,
                sequence=sequence,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return chunks


def is_block_boundary(line: str) -> bool:
    stripped = line.strip()
    return bool(
        not stripped
        or EMAIL_BOUNDARY_RE.match(stripped)
        or MESSAGE_SEPARATOR_RE.match(stripped)
    )


def split_blocks(text: str) -> list[tuple[str, int, int]]:
    text = normalize_text(text)
    if not text:
        return []

    blocks: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_start: int | None = None
    cursor = 0

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        line_start = cursor
        line_end = cursor + len(raw_line)
        cursor = line_end

        if is_block_boundary(line) and current:
            block_text = normalize_text("".join(current))
            if block_text and current_start is not None:
                blocks.append((block_text, current_start, line_start))
            current = []
            current_start = None

        if line.strip():
            if current_start is None:
                current_start = line_start
            current.append(raw_line)

    if current:
        block_text = normalize_text("".join(current))
        if block_text and current_start is not None:
            blocks.append((block_text, current_start, len(text)))

    return blocks


def split_oversized_block(block: tuple[str, int, int], config: ChunkingConfig) -> list[tuple[str, int, int]]:
    block_text, block_start, _block_end = block
    if len(block_text) <= config.block_max_chars:
        return [block]

    return [
        (chunk_text, block_start + start_char, block_start + end_char)
        for chunk_text, start_char, end_char in split_fixed_windows(
            block_text,
            config.max_chunk_chars,
            config.overlap_chars,
        )
    ]


def pack_blocks(
    blocks: list[tuple[str, int, int]],
    config: ChunkingConfig,
) -> list[tuple[str, int, int]]:
    packed: list[tuple[str, int, int]] = []
    current_texts: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_texts, current_start, current_end
        if not current_texts or current_start is None or current_end is None:
            return
        packed.append((normalize_text("\n\n".join(current_texts)), current_start, current_end))
        current_texts = []
        current_start = None
        current_end = None

    for raw_block in blocks:
        for block_text, block_start, block_end in split_oversized_block(raw_block, config):
            candidate_texts = [*current_texts, block_text]
            candidate = normalize_text("\n\n".join(candidate_texts))
            if current_texts and len(candidate) > config.block_max_chars:
                flush()

            if not current_texts:
                current_start = block_start
            current_texts.append(block_text)
            current_end = block_end

            current = normalize_text("\n\n".join(current_texts))
            if len(current) >= config.block_min_chars:
                flush()

    flush()
    return packed


def build_block_chunks_for_record(record: dict[str, Any], config: ChunkingConfig) -> list[dict[str, Any]]:
    source = str(record.get("source", ""))
    page = int(record.get("page") or 0)
    backend = str(record.get("backend", ""))
    text = record_text(record)

    blocks = split_blocks(text)
    if not blocks:
        return []

    chunks = []
    for sequence, (chunk_text, start_char, end_char) in enumerate(pack_blocks(blocks, config), start=1):
        chunks.append(
            make_chunk(
                strategy="chunk_block",
                source=source,
                page=page,
                backend=backend,
                sequence=sequence,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return chunks


def build_chunks_for_record(
    record: dict[str, Any],
    config: ChunkingConfig,
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    if strategy == "chunk_slide":
        return build_slide_chunks_for_record(record, config)
    if strategy == "chunk_block":
        return build_block_chunks_for_record(record, config)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def load_pages(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_corpus_strategy(
    pages_path: str | Path,
    *,
    strategy: str,
    output_path: str | Path | None = None,
    config: ChunkingConfig | None = None,
) -> Path:
    pages_path = Path(pages_path)
    chunking_config = config or ChunkingConfig()
    output = Path(output_path) if output_path else pages_path.with_name(
        strategy_output_filename(chunking_config, strategy)
    )

    chunk_count = 0
    with output.open("w", encoding="utf-8") as file:
        for record in load_pages(pages_path):
            for chunk in build_chunks_for_record(record, chunking_config, strategy=strategy):
                chunk_count += 1
                chunk["global_sequence"] = chunk_count
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log_block(
        "Index Build",
        "Chunk corpus",
        (
            f"Strategy: {strategy}\n"
            f"Chunks: {chunk_count}\n"
            f"Max chars: {chunking_config.max_chunk_chars}\n"
            f"Overlap chars: {chunking_config.overlap_chars}\n"
            f"Block target: {chunking_config.block_min_chars}-{chunking_config.block_max_chars}\n"
            f"Output: {output}"
        ),
    )
    return output


def chunk_corpus(
    pages_path: str | Path,
    *,
    output_path: str | Path | None = None,
    config: ChunkingConfig | None = None,
    strategy: str | None = None,
) -> Path:
    chunking_config = config or ChunkingConfig()
    selected_strategy = strategy or chunking_config.active_strategy
    return chunk_corpus_strategy(
        pages_path,
        strategy=selected_strategy,
        output_path=output_path,
        config=chunking_config,
    )


def chunk_corpus_all(
    pages_path: str | Path,
    *,
    config: ChunkingConfig | None = None,
) -> dict[str, Path]:
    chunking_config = config or ChunkingConfig()
    outputs = {}
    for strategy in chunking_config.strategies:
        outputs[strategy] = chunk_corpus_strategy(
            pages_path,
            strategy=strategy,
            config=chunking_config,
        )
    return outputs


def config_from_dict(config: dict[str, Any]) -> ChunkingConfig:
    chunking = config.get("chunking", {})
    strategies = chunking.get("strategies", ["chunk_slide", "chunk_block"])
    return ChunkingConfig(
        active_strategy=str(chunking.get("active_strategy", "chunk_block")),
        strategies=tuple(str(strategy) for strategy in strategies),
        max_chunk_chars=int(chunking.get("max_chunk_chars", 1200)),
        overlap_chars=int(chunking.get("overlap_chars", 400)),
        block_min_chars=int(chunking.get("block_min_chars", 1200)),
        block_max_chars=int(chunking.get("block_max_chars", 1600)),
        slide_output_filename=str(chunking.get("slide_output_filename", "chunks_slide.jsonl")),
        block_output_filename=str(chunking.get("block_output_filename", "chunks_block.jsonl")),
    )


def default_pages_path(config: dict[str, Any]) -> Path:
    parsing = config.get("parsing", {})
    backend = parsing.get("backend", "pdfplumber")
    output_dir = Path(parsing.get("output_dir", "parsed_corpus"))
    return output_dir / backend / "pages.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk parsed corpus pages.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pages", type=Path, help="Path to pages.jsonl. Defaults to config backend.")
    parser.add_argument("--output", type=Path, help="Path to write chunk JSONL.")
    parser.add_argument("--strategy", choices=sorted(STRATEGY_OUTPUTS), help="Chunking strategy to run.")
    parser.add_argument("--all", action="store_true", help="Generate every configured chunking strategy.")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    pages_path = args.pages or default_pages_path(raw_config)
    chunking_config = config_from_dict(raw_config)
    if args.all:
        if args.output:
            raise ValueError("--output cannot be used with --all")
        chunk_corpus_all(pages_path, config=chunking_config)
    else:
        chunk_corpus(
            pages_path,
            output_path=args.output,
            config=chunking_config,
            strategy=args.strategy,
        )


if __name__ == "__main__":
    main()
