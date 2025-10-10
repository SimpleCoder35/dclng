def md_sections_to_titled_dicts(md_text: str, min_level=2, max_level=2):
    pat = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
    matches = list(pat.finditer(md_text))
    out, idx = {}, 1
    for i, m in enumerate(matches):
        lvl = len(m.group(1))
        if not (min_level <= lvl <= max_level): 
            continue
        title = m.group(2).strip()
        start, end = m.start(), len(md_text)
        for j in range(i + 1, len(matches)):
            if len(matches[j].group(1)) <= lvl:
                end = matches[j].start(); break
        out[f"dict{idx}"] = f"{md_text[start:end].rstrip()}"
        out[f"dict{idx}_title"] = title      # optional: quick title access
        idx += 1
    return out


#### v2:
import re
from pathlib import Path
from typing import Dict

# match lines like: "## 1 Overview & Usage", "## 2 Model Information", etc.
MAIN_H2 = re.compile(r'^(##)\s+(\d+)\b.*$', re.MULTILINE)

def chunk_main_sections(md_text: str, include_heading: bool = True) -> Dict[str, str]:
    """
    Return {'dict1': '<section 1 text>', 'dict2': '<section 2 text>', ...}
    Only splits on level-2 headings that start with a number: '## 1', '## 2', ...
    """
    matches = list(MAIN_H2.finditer(md_text))
    out = {}
    for i, m in enumerate(matches, start=1):
        start = m.start() if include_heading else m.end()
        end = matches[i].start() if i < len(matches) else len(md_text)
        out[f"dict{i}"] = md_text[start:end].rstrip()
    return out

def chunk_main_sections_from_file(path: str, **kw) -> Dict[str, str]:
    return chunk_main_sections(Path(path).read_text(encoding="utf-8"), **kw)

# --- Example ---
# chunks = chunk_main_sections_from_file("doc.md")
# print(chunks.keys())         # dict1, dict2, ...
# print(chunks["dict1"][:400]) # preview
