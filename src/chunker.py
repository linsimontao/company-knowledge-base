"""Markdown-aware semantic chunker for enterprise policy documents with atomic clause isolation."""

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
    """Chunks Markdown documents respecting header hierarchy, table preservation, and atomic clause isolation."""

    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 120, atomic_clause_split: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.atomic_clause_split = atomic_clause_split

    def chunk_file(self, file_path: str | Path) -> List[DocumentChunk]:
        """Reads and chunks a single Markdown file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.chunk_text(content, source_file=path.name)

    def chunk_text(self, text: str, source_file: str = "unknown") -> List[DocumentChunk]:
        """Splits markdown text into semantic chunks with atomic clause boundaries."""
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

            # Identify clause domain (e.g. 年假, 病假, 事假, 报销, VPN, 试用期)
            clause_tag = self._extract_clause_tag(header_path, sec_content)

            # If section content fits within chunk_size, or atomic_clause_split is enabled for leaf sections
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
                            "clause_tag": clause_tag,
                            "char_count": len(chunk_text),
                        },
                    )
                )
                chunk_counter += 1
            else:
                # Sub-chunk section content while preserving tables and paragraph coherence
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
                                "clause_tag": clause_tag,
                                "char_count": len(chunk_text),
                            },
                        )
                    )
                    chunk_counter += 1

        return chunks

    def _extract_clause_tag(self, header_path: str, content: str) -> str:
        """Extracts specific business clause tags for filtering and targeted retrieval."""
        combined = f"{header_path} {content}"
        tags = [
            ("年假", "年休假"),
            ("病假", "带薪病假"),
            ("事假", "无薪事假"),
            ("育儿假", "育儿假"),
            ("婚假", "产假"),
            ("报销", "发票"),
            ("VPN", "远程接入"),
            ("试用期", "转正"),
            ("远程办公", "混合办公"),
            ("公积金", "薪酬"),
            ("离职", "交接"),
            ("信息安全", "机密"),
        ]
        for tag, keyword in tags:
            if tag in combined or keyword in combined:
                return tag
        return "通用条款"

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
                if level == 1:
                    current_headers = [header_text]
                elif level == 2:
                    current_headers = current_headers[:1] + [header_text]
                elif level == 3:
                    current_headers = current_headers[:2] + [header_text]
                elif level == 4:
                    current_headers = current_headers[:3] + [header_text]
            else:
                current_lines.append(line)

        # Append final section
        if current_lines:
            sections.append(
                {
                    "headers": list(current_headers),
                    "lines": list(current_lines),
                }
            )

        return sections

    def _split_text_with_overlap(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text recursively using paragraphs, list items, or sentences while keeping markdown tables intact."""
        # Check if text is a single table
        if self._is_markdown_table(text) and len(text) <= chunk_size * 1.5:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(current_chunk) + len(p) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{p}".strip() if current_chunk else p
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(p) <= chunk_size:
                    # Start new chunk with overlap if feasible
                    if chunk_overlap > 0 and chunks:
                        overlap_tail = chunks[-1][-chunk_overlap:]
                        current_chunk = f"...{overlap_tail}\n\n{p}".strip()
                    else:
                        current_chunk = p
                else:
                    # Paragraph itself exceeds chunk_size, split by list items or sentences
                    sub_splits = self._split_large_paragraph(p, chunk_size, chunk_overlap)
                    for s in sub_splits:
                        chunks.append(s)
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _is_markdown_table(self, text: str) -> bool:
        """Checks whether the text represents a Markdown table."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2 and "|" in lines[0] and re.match(r"^\|?\s*[-:]+[-| :]*\|?$", lines[1]):
            return True
        return False

    def _split_large_paragraph(self, p: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits an oversized paragraph by bullet items or punctuation."""
        items = re.split(r"(?<=\n)(?=[-*\d]+\.\s+)", p)
        if len(items) > 1:
            res = []
            cur = ""
            for it in items:
                if len(cur) + len(it) <= chunk_size:
                    cur = f"{cur}\n{it}".strip() if cur else it
                else:
                    if cur:
                        res.append(cur)
                    cur = it
            if cur:
                res.append(cur)
            return res

        # Sentence punctuation split
        sentences = re.split(r"(?<=[。；;！？\n])", p)
        res = []
        cur = ""
        for s in sentences:
            if not s:
                continue
            if len(cur) + len(s) <= chunk_size:
                cur += s
            else:
                if cur:
                    res.append(cur)
                cur = s
        if cur:
            res.append(cur)
        return res if res else [p]
