#!/usr/bin/env python3
"""Convert uploaded files (PDF, Word, Excel/CSV, ARM exports) to Markdown.

Usage: dossier-rag convert <file|folder|.zip> [...] --out <dir>

For each input writes <dir>/<stem>.md and, for every table found,
<dir>/<stem>_tables/table_NN.csv (the CSVs are the source of truth for numbers).
"""
import argparse, os, re, subprocess, sys, shutil
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_md(df):
    df = df.fillna("").astype(str)
    df = df.apply(lambda c: c.str.replace("\n", " ").str.replace("|", "\\|").str.strip())
    cols = [str(c).replace("\n", " ").replace("|", "\\|").strip() for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


LIGS = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}


def fix_ligs(s):
    for k, v in LIGS.items():
        s = s.replace(k, v)
    return s


def save_table(df, tdir, n):
    df = df.map(lambda x: fix_ligs(x) if isinstance(x, str) else x)
    df.columns = [fix_ligs(str(c)) for c in df.columns]
    tdir.mkdir(parents=True, exist_ok=True)
    p = tdir / f"table_{n:02d}.csv"
    df.to_csv(p, index=False)
    return p


def header_promote(df):
    """Camelot gives integer columns; use the first row as the header."""
    if len(df) > 1 and all(isinstance(c, (int, np.integer)) for c in df.columns):
        hdr = [h if h.strip() else f"col{i+1}" for i, h in enumerate(df.iloc[0].astype(str))]
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = hdr
    return df


def page_texts(path, npages):
    """Column-aware text via poppler's pdftotext; pdfplumber as fallback."""
    if shutil.which("pdftotext"):
        out = []
        for i in range(1, npages + 1):
            r = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(path), "-"],
                               capture_output=True, text=True)
            out.append(r.stdout)
        return out
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [(p.extract_text(x_tolerance=1.5) or "") for p in pdf.pages]


def caption_areas(path):
    """Find 'Table N.' captions and the rule-bounded area under each.
    Returns {page: [(caption_text, 'x1,y1,x2,y2' in PDF points)]} for Camelot."""
    import pdfplumber
    areas = {}
    with pdfplumber.open(path) as pdf:
        for pn, p in enumerate(pdf.pages, 1):
            caps = [l for l in p.extract_text_lines(x_tolerance=1)
                    if re.match(r"^Table\s*\d+[.:]", l["text"])]
            if not caps:
                continue
            rules = [l for l in p.lines + p.edges
                     if abs(l["top"] - l["bottom"]) < 1.5 and l["top"] > 40 and l["top"] < p.height - 40]
            caps.sort(key=lambda c: c["top"])
            for k, c in enumerate(caps):
                nxt = caps[k + 1]["top"] if k + 1 < len(caps) else p.height
                below = [r for r in rules if c["top"] < r["top"] < nxt]
                if not below:
                    continue
                # keep only rules about as wide as the widest one (drops footnote rules)
                wmax = max(r["x1"] - r["x0"] for r in below)
                below = sorted([r for r in below if (r["x1"] - r["x0"]) > 0.8 * wmax],
                               key=lambda r: r["top"])
                # distinct rule positions (thick rules are drawn twice)
                ys = []
                for r in below:
                    if not ys or r["top"] - ys[-1] > 3:
                        ys.append(r["top"])
                if len(ys) < 2:
                    continue
                x0 = min(r["x0"] for r in below); x1 = max(r["x1"] for r in below)
                top, bot = ys[0] - 1, ys[-1] + 1
                # column boundaries = gaps in the header (text between first two rules)
                hdr = [w for w in p.extract_words(x_tolerance=1) if ys[0] < w["top"] < ys[1] and x0 <= w["x0"] <= x1]
                spans = []
                for w in sorted(hdr, key=lambda w: w["x0"]):
                    if spans and w["x0"] - spans[-1][1] < 10:
                        spans[-1][1] = max(spans[-1][1], w["x1"])
                    else:
                        spans.append([w["x0"], w["x1"]])
                cols = [f"{b[0] - 2:.0f}" for a, b in zip(spans, spans[1:])]  # left-aligned columns
                areas.setdefault(pn, []).append(
                    (c["text"], f"{x0:.0f},{p.height - top:.0f},{x1:.0f},{p.height - bot:.0f}",
                     ",".join(cols)))
    return areas


def merge_wrapped(df):
    """Stream mode makes one row per text line. A row whose first cell is empty
    is a wrapped continuation of the row above, so join it upward."""
    rows = []
    for r in df.astype(str).values.tolist():
        first = r[0].strip()
        cont = rows and ((not first and any(c.strip() for c in r[1:]))
                         or (len(r) == 1 and first[:1].islower() and len(rows) > 1))
        if cont:
            rows[-1] = [(a + " " + b).strip() for a, b in zip(rows[-1], r)]
        else:
            rows.append(r)
    # two-line header: if the header row has an empty cell, the next row is its 2nd line
    if len(rows) > 2 and any(not c.strip() for c in rows[0]):
        rows[0] = [(a + " " + b).strip() for a, b in zip(rows[0], rows[1])]
        del rows[1]
    return pd.DataFrame(rows)


def convert_pdf(path, out_md, tdir, notes):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        npages = len(pdf.pages)
    pages = page_texts(path, npages)
    if sum(len(t.strip()) for t in pages) < 50 * max(1, len(pages)) * 0.2:
        notes.append("Little or no text layer: probably a scanned PDF.")
        if shutil.which("ocrmypdf"):
            ocr = tdir.parent / (path.stem + "_ocr.pdf")
            subprocess.run(["ocrmypdf", "--skip-text", str(path), str(ocr)], check=False,
                           capture_output=True)
            if ocr.exists():
                notes.append("OCR applied with ocrmypdf; check all values carefully.")
                return convert_pdf(ocr, out_md, tdir, notes)
        notes.append("ocrmypdf not available: text/tables could not be extracted.")

    import camelot
    tables_by_page = {}
    n = 0

    def add(t, flavor, caption=""):
        nonlocal n
        n += 1
        df = header_promote(t.df.copy())
        csv = save_table(df, tdir, n)
        acc = t.parsing_report.get("accuracy", 0)
        tables_by_page.setdefault(int(t.page), []).append((n, flavor, acc, csv, df, caption))

    # 1) Fully ruled (grid) tables anywhere.
    lattice_pages = set()
    try:
        for t in camelot.read_pdf(str(path), pages="all", flavor="lattice"):
            if t.df.shape[0] > 1 and t.df.shape[1] > 1:
                add(t, "lattice"); lattice_pages.add(int(t.page))
    except Exception as e:
        notes.append(f"Camelot lattice failed: {e}")
    # 2) Tables with horizontal rules only (journal/report style): stream mode,
    #    limited to the area under each 'Table N.' caption. Running stream on whole
    #    pages turns multi-column body text into fake tables, so we never do that.
    for pn, items in caption_areas(path).items():
        if pn in lattice_pages:
            continue
        for caption, area, cols in items:
            kw = dict(table_areas=[area])
            if cols:
                kw["columns"] = [cols]
            try:
                tl = camelot.read_pdf(str(path), pages=str(pn), flavor="stream", **kw)
            except Exception as e:
                notes.append(f"Camelot stream failed on page {pn}: {e}")
                continue
            for t in tl:
                t.df = merge_wrapped(t.df)
                if t.df.shape[0] > 1:
                    add(t, "stream", caption)
    if any("(cid:" in t for t in pages) or any(
            "(cid:" in df.to_string() for v in tables_by_page.values() for (_, _, _, _, df, _) in v):
        notes.append("Some symbols could not be decoded and appear as (cid:N), often ± or ≤/≥; "
                     "check them against the PDF and replace them.")
    if not tables_by_page and any(re.search(r"\bTable\s*\d", t) for t in pages):
        notes.append("Text mentions tables but none could be extracted; check the PDF manually.")
    out = []
    for i, text in enumerate(pages, 1):
        out.append(f"\n<!-- page {i} -->\n")
        out.append(text.strip())
        for (k, flavor, acc, csv, df, cap) in tables_by_page.get(i, []):
            label = f"**{cap}**" if cap else f"**Extracted table {k}**"
            out.append(f"\n{label}\n\n_(extracted table {k}, page {i}, camelot {flavor}, "
                       f"accuracy {acc:.0f}%, source: `{csv.relative_to(out_md.parent)}`)_\n")
            out.append(df_to_md(df))
    if n:
        notes.append(f"{n} table(s) extracted with Camelot. Table text also appears in the page text above it.")
    return "\n".join(out)


def convert_docx(path, out_md, tdir, notes):
    media = tdir.parent / (path.stem + "_media")
    r = subprocess.run(["pandoc", str(path), "-t", "gfm", "--wrap=none",
                        f"--extract-media={media}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    try:  # tables also as CSV
        from docx import Document
        for k, t in enumerate(Document(str(path)).tables, 1):
            rows = [[c.text.strip() for c in row.cells] for row in t.rows]
            if len(rows) > 1:
                save_table(pd.DataFrame(rows[1:], columns=rows[0]), tdir, k)
    except Exception:
        pass
    if "<table>" in r.stdout:
        notes.append("Some Word tables have merged cells and were kept as HTML tables.")
    return r.stdout


def convert_sheet(path, out_md, tdir, notes):
    out, n = [], 0
    if path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else None
        sheets = {"data": pd.read_csv(path, sep=sep, engine="python")}
    else:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        if isinstance(df.columns[0], (int, np.integer)):  # raw sheet: first non-empty row = header
            df.columns = [str(c) if pd.notna(c) else f"col{i+1}"
                          for i, c in enumerate(df.iloc[0])]
            df = df.iloc[1:].reset_index(drop=True)
        n += 1
        csv = save_table(df, tdir, n)
        out.append(f"## Sheet: {name}\n\n_{len(df)} rows × {len(df.columns)} columns; "
                   f"source: `{csv.relative_to(out_md.parent)}`_\n")
        out.append(df_to_md(df.head(500)))
        if len(df) > 500:
            notes.append(f"Sheet '{name}' has {len(df)} rows; Markdown shows the first 500, the CSV has all.")
    return "\n\n".join(out)


def convert(path, outdir):
    path = Path(path)
    out_md = outdir / (path.stem + ".md")
    tdir = outdir / (path.stem + "_tables")
    notes = []
    ext = path.suffix.lower()
    if ext == ".pdf":
        body = convert_pdf(path, out_md, tdir, notes)
    elif ext in (".docx", ".odt", ".rtf", ".doc"):
        if ext == ".doc":
            raise RuntimeError(".doc: convert to .docx first (e.g. soffice --headless --convert-to docx)")
        body = convert_docx(path, out_md, tdir, notes)
    elif ext in (".xlsx", ".xlsm", ".xls", ".ods", ".csv", ".tsv"):
        body = convert_sheet(path, out_md, tdir, notes)
    elif ext in (".md", ".txt"):
        body = path.read_text(errors="replace")
    else:
        raise RuntimeError(f"Unsupported file type {ext}")
    head = ["---", f"source_file: {path.name}", f"source_type: {ext.lstrip('.')}",
            "status: unverified  # check tables against the original before use"]
    if notes:
        head.append("conversion_notes:")
        head += [f"  - {quote(n)}" for n in notes]
    head.append("---\n")
    out_md.write_text("\n".join(head) + f"# {path.stem}\n\n" + fix_ligs(body).strip() + "\n")
    return out_md, notes


def quote(s):
    return '"' + s.replace('"', "'") + '"'


SUPPORTED = {".pdf", ".docx", ".odt", ".rtf", ".doc", ".xlsx", ".xlsm", ".xls", ".ods",
             ".csv", ".tsv", ".md", ".txt"}
ARCHIVES = {".zip"}


def safe_extract(zpath, dest):
    """Extract a zip into an empty folder, refusing paths that escape it."""
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe path in archive: {m.filename}")
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return dest


def collect(inputs, outdir):
    """Expand folders and zip archives into (file, relative_output_dir) pairs."""
    jobs, skipped = [], []

    def walk(base, rel):
        for f in sorted(base.rglob("*")):
            if not f.is_file() or any(part.startswith((".", "__MACOSX")) for part in f.relative_to(base).parts):
                continue
            sub = rel / f.relative_to(base).parent
            ext = f.suffix.lower()
            if ext in ARCHIVES:
                ex = outdir / "_extracted" / sub / f.stem
                walk(safe_extract(f, ex), sub / f.stem)
            elif ext in SUPPORTED:
                jobs.append((f, sub))
            else:
                skipped.append(str(f))

    for i in inputs:
        p = Path(i)
        if p.is_dir():
            walk(p, Path(p.name))
        elif p.suffix.lower() in ARCHIVES:
            walk(safe_extract(p, outdir / "_extracted" / p.stem), Path(p.stem))
        elif p.suffix.lower() in SUPPORTED:
            jobs.append((p, Path(".")))
        else:
            skipped.append(str(p))
    return jobs, skipped


def convert_all(inputs, outdir, log=print):
    """Convert files/folders/zips. Returns (results, skipped); each result is a dict
    with src, md (Path or None), notes, error."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    jobs, skipped = collect(inputs, outdir)
    results = []
    for f, sub in jobs:
        dest = outdir / sub
        dest.mkdir(parents=True, exist_ok=True)
        try:
            md, notes = convert(f, dest)
            results.append(dict(src=f, md=md, notes=notes, error=None))
            log(f"OK   {f} -> {md}" + "".join(f"\n     note: {n}" for n in notes))
        except Exception as e:
            results.append(dict(src=f, md=None, notes=[], error=str(e)))
            log(f"FAIL {f}: {e}")
    for s_ in skipped:
        log(f"SKIP {s_}: unsupported type")
    return results, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dossier-rag convert",
                                 description="Convert files, folders and .zip archives to Markdown + table CSVs.")
    ap.add_argument("inputs", nargs="+", help="files, folders or .zip archives")
    ap.add_argument("--out", default="converted")
    a = ap.parse_args(argv)
    results, skipped = convert_all(a.inputs, a.out)
    ok = sum(1 for r in results if r["md"])
    print(f"\n{ok} converted, {len(results) - ok} failed, {len(skipped)} skipped.")
    return 1 if results and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
