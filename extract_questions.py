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
