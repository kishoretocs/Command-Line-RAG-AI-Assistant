from collections import defaultdict
from typing import List, Dict, Any, Tuple

class ContextBuilder:
    @staticmethod
    def build_context(section_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assembles deduplicated section chunks into an organized, readable context block.
        Groups by document_id and injects temporal/superseded status headers.
        Returns:
          1. assembled_context_str
          2. citations_meta: List of available document/section sources
        """
        if not section_chunks:
            return "No relevant documentation found.", []

        # Group by document_id
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sec in section_chunks:
            grouped[sec["document_id"]].append(sec)

        blocks: List[str] = []
        citations_meta: List[Dict[str, Any]] = []

        for doc_id, sections in grouped.items():
            first = sections[0]
            meta = first.get("metadata", {})
            title = first.get("title", doc_id)
            version = meta.get("version", "1.0")
            effective = meta.get("effective_date", "Unknown")
            is_superseded = meta.get("is_superseded", False)
            classification = meta.get("classification", "Internal")
            owner = meta.get("owner", "")

            status_str = (
                "SUPERSEDED (older version — still valid, authoritative content; "
                "prefer a newer version only if this context also includes one)"
            ) if is_superseded else "CURRENT / ACTIVE"

            header = (
                f"=== DOCUMENT: {title} ({doc_id}) ===\n"
                f"Status: {status_str} | Version: {version} | "
                f"Effective Date: {effective} | Classification: {classification} | Owner: {owner}\n"
            )

            # Sort sections by section_number
            sorted_sections = sorted(sections, key=lambda s: str(s.get("section_number", "0")))

            section_texts = []
            for sec in sorted_sections:
                sec_num = sec.get("section_number", "")
                sec_title = sec.get("section_title", "")
                p_start = sec.get("page_start", sec.get("metadata", {}).get("page", 1))
                p_end = sec.get("page_end", p_start)
                page_str = f"page {p_start}" if p_start == p_end else f"pages {p_start}-{p_end}"

                sec_header = f"[Section {sec_num}: {sec_title} — {page_str}]\n"
                # Retriever hit dicts carry the body under "text" (context-injected variant);
                # raw section chunks carry it under "full_text". Accept both shapes.
                sec_content = (
                    sec.get("full_text")
                    or sec.get("text")
                    or sec.get("raw_text", "")
                ).strip()
                # Drop the redundant injected "[Document: ...]" header line if present
                if sec_content.startswith("[Document:"):
                    sec_content = sec_content.split("\n", 1)[1].strip() if "\n" in sec_content else ""
                section_texts.append(sec_header + sec_content)

                citations_meta.append({
                    "document_id": doc_id,
                    "section": sec_num,
                    "page": p_start,
                    "version": version
                })

            doc_block = header + "\n" + "\n\n".join(section_texts)
            blocks.append(doc_block)

        assembled_context_str = "\n\n" + ("=" * 50) + "\n\n".join(blocks)
        return assembled_context_str, citations_meta
