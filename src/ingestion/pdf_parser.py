from pathlib import Path
from typing import List, Dict, Any

class PDFParser:
    def __init__(self):
        pass

    def parse_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Parses a PDF into structured pages.
        Extracts markdown text with tables preserved in markdown syntax (| col | col |).
        Uses pdfplumber as primary extractor with fallback to pypdf.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        pages: List[Dict[str, Any]] = []

        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    page_text = page.extract_text() or ""
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    table_md_blocks = []
                    if tables:
                        for table in tables:
                            if not table or len(table) < 1:
                                continue
                            # Clean rows
                            clean_rows = []
                            for row in table:
                                clean_rows.append([str(c or "").strip().replace("\n", " ") for c in row])
                            
                            if clean_rows:
                                header = " | ".join(clean_rows[0])
                                sep = " | ".join(["---"] * len(clean_rows[0]))
                                md_lines = [f"| {header} |", f"| {sep} |"]
                                for r in clean_rows[1:]:
                                    md_lines.append(f"| {' | '.join(r)} |")
                                table_md_blocks.append("\n".join(md_lines))

                    # Combine tables and text
                    combined_text = page_text.strip()
                    if table_md_blocks:
                        combined_text += "\n\n" + "\n\n".join(table_md_blocks)

                    pages.append({
                        "page": page_num,
                        "text": combined_text
                    })
            return pages

        except Exception as e:
            # Fallback to pypdf
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for idx, page in enumerate(reader.pages):
                pages.append({
                    "page": idx + 1,
                    "text": (page.extract_text() or "").strip()
                })
            return pages
