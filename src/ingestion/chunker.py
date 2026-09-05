import re
from typing import List, Dict, Any, Tuple
import tiktoken
from src.config import MAX_SECTION_TOKENS

# Tokenizer helper
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        # Fallback approximation: 1 token ≈ 0.75 words
        return max(1, int(len(text.split()) * 1.3))


class SectionChunker:
    def __init__(
        self,
        max_section_tokens: int = MAX_SECTION_TOKENS
    ):
        self.max_section_tokens = max_section_tokens

    def _split_into_sections(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses pages into discrete semantic sections based on section headers (e.g. '1. Purpose', '§2').
        """
        sections: List[Dict[str, Any]] = []
        current_sec = {
            "section_number": "0",
            "section_title": "Overview / Preamble",
            "blocks": [],
            "page_start": 1,
            "page_end": 1
        }

        # Regex for common section headers in policies ("1. Purpose", "4.6 Sick leave", "## 2. Scope")
        header_regex = re.compile(r"^(?:#{1,3}\s+)?(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s,&/\-–—]+)$", re.MULTILINE)

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]

            # Process line-by-line: headers often appear mid-block after single newlines
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                match = header_regex.match(line)
                if match:
                    sec_num = match.group(1)
                    sec_title = match.group(2).strip()

                    # Guards: real section numbers are small (1-99); real titles are short.
                    # Rejects table values ("1000 Enterprise") and numbered list items
                    # ("1. Report a suspected incident to the ... within one hour").
                    if int(sec_num.split(".")[0]) > 99 or len(sec_title) > 60 or len(sec_title.split()) > 10:
                        current_sec["blocks"].append(line)
                        current_sec["page_end"] = page_num
                        continue

                    # Finalize previous section if it has content
                    if current_sec["blocks"]:
                        sections.append(current_sec)

                    current_sec = {
                        "section_number": sec_num,
                        "section_title": sec_title,
                        "blocks": [line],
                        "page_start": page_num,
                        "page_end": page_num
                    }
                else:
                    current_sec["blocks"].append(line)
                    current_sec["page_end"] = page_num

        if current_sec["blocks"]:
            sections.append(current_sec)

        return sections

    def _split_oversized_section(self, full_text: str, blocks: List[str], sec_title: str) -> List[Tuple[str, str]]:
        """
        If a section > max_section_tokens, split into Part 1, Part 2 at paragraph boundaries.
        Returns list of (part_title, part_text).
        """
        parts = []
        current_part_blocks = []
        current_part_tokens = 0
        part_idx = 1

        for block in blocks:
            b_tokens = count_tokens(block)
            if current_part_tokens + b_tokens > self.max_section_tokens and current_part_blocks:
                # Emit current part
                part_text = "\n\n".join(current_part_blocks)
                parts.append((f"{sec_title} (Part {part_idx})", part_text))
                part_idx += 1
                current_part_blocks = [block]
                current_part_tokens = b_tokens
            else:
                current_part_blocks.append(block)
                current_part_tokens += b_tokens

        if current_part_blocks:
            part_text = "\n\n".join(current_part_blocks)
            if parts:
                parts.append((f"{sec_title} (Part {part_idx})", part_text))
            else:
                parts.append((sec_title, part_text))

        return parts

    def chunk_document(
        self,
        doc_meta: Dict[str, Any],
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Takes parsed pages of a document and returns a list of section chunks
        (single-level direct section chunking with context headers).
        """
        doc_id = doc_meta["document_id"]
        title = doc_meta["title"]
        version = doc_meta["version"]
        effective_date = doc_meta["effective_date"]
        is_superseded = doc_meta.get("is_superseded", False)

        sections = self._split_into_sections(pages)
        section_chunks: List[Dict[str, Any]] = []
        used_ids: set = set()

        for sec in sections:
            sec_num = sec["section_number"]
            sec_title = sec["section_title"]
            blocks = sec["blocks"]
            full_text = "\n\n".join(blocks)
            total_tokens = count_tokens(full_text)

            # Determine section parts if oversized
            if total_tokens > self.max_section_tokens:
                sec_parts = self._split_oversized_section(full_text, blocks, sec_title)
            else:
                sec_parts = [(sec_title, full_text)]

            for part_idx, (p_title, p_text) in enumerate(sec_parts, 1):
                chunk_id = f"{doc_id}_sec_{sec_num}" if len(sec_parts) == 1 else f"{doc_id}_sec_{sec_num}_p{part_idx}"

                # Safety net: guarantee unique chunk IDs per document
                base_id, dup_n = chunk_id, 1
                while chunk_id in used_ids:
                    dup_n += 1
                    chunk_id = f"{base_id}_d{dup_n}"
                used_ids.add(chunk_id)

                header = (
                    f"[Document: {doc_id} {title} | Section: {p_title} | "
                    f"Version: {version} | Effective: {effective_date}]\n"
                )
                context_injected = header + p_text

                chunk_obj = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "title": title,
                    "section_number": sec_num,
                    "section_title": p_title,
                    "part_index": part_idx,
                    "total_parts": len(sec_parts),
                    "page_start": sec["page_start"],
                    "page_end": sec["page_end"],
                    "token_count": count_tokens(context_injected),
                    "context_injected_text": context_injected,
                    "full_text": p_text,
                    "metadata": {
                        "document_id": doc_id,
                        "section_number": sec_num,
                        "section_title": p_title,
                        "page": sec["page_start"],
                        "version": version,
                        "effective_date": effective_date,
                        "is_superseded": is_superseded,
                        "classification": doc_meta.get("classification", "Internal")
                    }
                }
                section_chunks.append(chunk_obj)

        return section_chunks
