"""Markdown-aware semantic chunker for enterprise policy documents."""

import re
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunk of document with enriched metadata."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="The text content of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata including source, section, headers"
    )


class MarkdownChunker:
    """Chunks Markdown documents respecting header hierarchy and semantic boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_file(self, file_path: str | Path) -> List[DocumentChunk]:
        """Reads and chunks a single Markdown file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.chunk_text(content, source_file=path.name)

    def chunk_text(self, text: str, source_file: str = "unknown") -> List[DocumentChunk]:
        """Splits markdown text into semantic chunks with metadata."""
        lines = text.splitlines()

        # Extract document title (first # header if present)
        doc_title = source_file
        for line in lines:
            if line.strip().startswith("# "):
                doc_title = line.strip().lstrip("#").strip()
                break

        # Parse sections based on markdown headers
        sections = self._parse_markdown_sections(lines, doc_title)

        chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for sec in sections:
            header_path = " > ".join(sec["headers"])
            sec_content = "\n".join(sec["lines"]).strip()

            if not sec_content:
                continue

            # If section content is within chunk_size, make it a single chunk
            if len(sec_content) <= self.chunk_size:
                chunk_id = f"{source_file}#chunk_{chunk_counter}"
                chunk_text = f"[{header_path}]\n{sec_content}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={
                            "source": source_file,
                            "title": doc_title,
                            "section": sec["headers"][-1] if sec["headers"] else doc_title,
                            "header_path": header_path,
                            "chunk_index": chunk_counter,
                            "char_count": len(chunk_text),
                        },
                    )
                )
                chunk_counter += 1
            else:
                # Sub-chunk the section content using recursive sliding window
                sub_texts = self._split_text_with_overlap(
                    sec_content, self.chunk_size, self.chunk_overlap
                )
                for sub_idx, sub_t in enumerate(sub_texts):
                    chunk_id = f"{source_file}#chunk_{chunk_counter}"
                    chunk_text = f"[{header_path} (Part {sub_idx+1})]\n{sub_t}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            metadata={
                                "source": source_file,
                                "title": doc_title,
                                "section": sec["headers"][-1] if sec["headers"] else doc_title,
                                "header_path": header_path,
                                "chunk_index": chunk_counter,
                                "sub_index": sub_idx,
                                "char_count": len(chunk_text),
                            },
                        )
                    )
                    chunk_counter += 1

        return chunks

    def _parse_markdown_sections(self, lines: List[str], doc_title: str) -> List[Dict[str, Any]]:
        """Splits markdown lines into sections according to headers (#, ##, ###, ####)."""
        sections = []
        current_headers = [doc_title]
        current_lines = []

        header_regex = re.compile(r"^(#{1,4})\s+(.*)$")

        for line in lines:
            match = header_regex.match(line)
            if match:
                # Save previous section if it has content
                if current_lines:
                    sections.append(
                        {
                            "headers": list(current_headers),
                            "lines": list(current_lines),
                        }
                    )
                    current_lines = []

                level = len(match.group(1))
                header_text = match.group(2).strip()

                # Adjust header hierarchy stack
                # If level 1 (# Document Title)
                if level == 1:
                    current_headers = [header_text]
                elif level == 2:
                    current_headers = [current_headers[0], header_text]
                elif level == 3:
                    base = current_headers[:2] if len(current_headers) >= 2 else current_headers
                    current_headers = base + [header_text]
                elif level == 4:
                    base = current_headers[:3] if len(current_headers) >= 3 else current_headers
                    current_headers = base + [header_text]
            else:
                current_lines.append(line)

        # Append last section
        if current_lines:
            sections.append(
                {
                    "headers": list(current_headers),
                    "lines": list(current_lines),
                }
            )

        return sections

    def _split_text_with_overlap(
        self, text: str, max_size: int, overlap: int
    ) -> List[str]:
        """Splits a long string into overlapping segments using semantic delimiters."""
        if len(text) <= max_size:
            return [text]

        # Priority delimiters: paragraph -> list/newline -> punctuation -> comma -> whitespace
        delimiters = ["\n\n", "\n", "。\n", "；\n", "。", "；", "！", "？", "，", " "]
        
        # Helper to split by primary delimiter
        paragraphs = re.split(r"(\n\n+|\n(?=[0-9]+\.|\-|\*))", text)
        
        chunks = []
        current_chunk = ""

        for piece in paragraphs:
            if not piece:
                continue
            if len(current_chunk) + len(piece) <= max_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap from the end of current_chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_prefix = current_chunk[-overlap:]
                    # Try to break at a clean punctuation or newline
                    clean_start = max(
                        overlap_prefix.rfind("\n"),
                        overlap_prefix.rfind("。"),
                        overlap_prefix.rfind("；"),
                    )
                    if clean_start != -1:
                        overlap_prefix = overlap_prefix[clean_start + 1 :]
                    current_chunk = overlap_prefix + piece
                else:
                    current_chunk = piece

                # If a single piece is still excessively large (larger than max_size)
                while len(current_chunk) > max_size:
                    split_idx = max_size
                    # look for last delimiter before max_size
                    for delim in ["\n", "。", "；", "，", " "]:
                        idx = current_chunk.rfind(delim, 0, max_size)
                        if idx > max_size // 2:
                            split_idx = idx + len(delim)
                            break
                    
                    chunks.append(current_chunk[:split_idx].strip())
                    current_chunk = current_chunk[split_idx - overlap if split_idx > overlap else split_idx :]

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

