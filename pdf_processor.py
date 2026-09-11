"""
PDF Processor Module for RAG Document Intelligence.
Extracts text and metadata from PDF documents while preserving page boundaries.
"""

import os
import io
import re
from typing import List, Dict, Any, Optional

class PDFProcessor:
    def __init__(self):
        self.supported_extensions = [".pdf", ".txt", ".md"]

    def extract_text_from_file(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF or text bytes.
        Returns a list of page dictionaries:
        [{"page": int, "text": str, "source": str, "char_count": int, "word_count": int}]
        """
        ext = os.path.splitext(filename)[1].lower()
        pages = []

        if ext == ".pdf":
            # Attempt to use pypdf or PyPDF2 if installed
            pdf_extracted = False
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    clean_text = self._clean_text(text)
                    if clean_text:
                        pages.append({
                            "page": idx + 1,
                            "text": clean_text,
                            "source": filename,
                            "char_count": len(clean_text),
                            "word_count": len(clean_text.split())
                        })
                pdf_extracted = True
            except ImportError:
                pass
            except Exception as e:
                print(f"pypdf extraction error: {e}")

            if not pdf_extracted:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for idx in range(len(doc)):
                        page = doc[idx]
                        text = page.get_text() or ""
                        clean_text = self._clean_text(text)
                        if clean_text:
                            pages.append({
                                "page": idx + 1,
                                "text": clean_text,
                                "source": filename,
                                "char_count": len(clean_text),
                                "word_count": len(clean_text.split())
                            })
                    pdf_extracted = True
                except ImportError:
                    pass

            if not pdf_extracted and not pages:
                # Fallback text extraction if PDF libraries are not yet compiled/installed
                try:
                    raw_str = file_bytes.decode("utf-8", errors="ignore")
                    # Try splitting on form feeds or page markers
                    raw_pages = raw_str.split("\x0c")
                    if len(raw_pages) <= 1:
                        raw_pages = [raw_str]
                    for idx, pt in enumerate(raw_pages):
                        clean_text = self._clean_text(pt)
                        if clean_text:
                            pages.append({
                                "page": idx + 1,
                                "text": clean_text,
                                "source": filename,
                                "char_count": len(clean_text),
                                "word_count": len(clean_text.split())
                            })
                except Exception:
                    pass
        else:
            # Plain text / Markdown
            text = file_bytes.decode("utf-8", errors="ignore")
            pages = self._split_by_page_markers(text, filename)

        # Secondary check: if only 1 page was extracted, check if page markers exist inside the text
        if len(pages) == 1 and pages[0].get("text"):
            re_split = self._split_by_page_markers(pages[0]["text"], filename)
            if len(re_split) > 1:
                pages = re_split

        return pages

    def _split_by_page_markers(self, raw_text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Splits text by common page markers (e.g., '--- Page 1 ---', 'Page 1', '\x0c', '=== Page 1 ===').
        """
        marker_pattern = re.compile(
            r'(?:^|\n)\s*(?:---|===)?\s*(?:Page|PAGE)\s+(\d+)(?:\s*(?:of|OF|\/)\s*\d+)?\s*(?:---|===)?\s*\n',
            re.IGNORECASE
        )
        matches = list(marker_pattern.finditer(raw_text))
        
        pages = []
        if len(matches) > 1 or (len(matches) == 1 and matches[0].start() < 100):
            for i, m in enumerate(matches):
                p_num = int(m.group(1))
                start_pos = m.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
                p_text = self._clean_text(raw_text[start_pos:end_pos])
                if p_text:
                    pages.append({
                        "page": p_num,
                        "text": p_text,
                        "source": filename,
                        "char_count": len(p_text),
                        "word_count": len(p_text.split())
                    })
            if pages:
                return pages

        # Fallback to form feed delimiters
        if "\x0c" in raw_text:
            raw_pages = raw_text.split("\x0c")
            for idx, pt in enumerate(raw_pages):
                clean_text = self._clean_text(pt)
                if clean_text:
                    pages.append({
                        "page": idx + 1,
                        "text": clean_text,
                        "source": filename,
                        "char_count": len(clean_text),
                        "word_count": len(clean_text.split())
                    })
            if pages:
                return pages

        # Fallback to horizontal rule dividers
        paragraphs = re.split(r'\n\s*---\s*\n', raw_text)
        if len(paragraphs) > 1:
            for idx, p in enumerate(paragraphs):
                clean_text = self._clean_text(p)
                if clean_text:
                    pages.append({
                        "page": idx + 1,
                        "text": clean_text,
                        "source": filename,
                        "char_count": len(clean_text),
                        "word_count": len(clean_text.split())
                    })
            if pages:
                return pages

        # Single page fallback
        clean_text = self._clean_text(raw_text)
        if clean_text:
            pages.append({
                "page": 1,
                "text": clean_text,
                "source": filename,
                "char_count": len(clean_text),
                "word_count": len(clean_text.split())
            })
        return pages

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
