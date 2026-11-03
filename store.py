"""Search index + trial-table database in a single SQLite file.

- docs:      one row per converted source file
- chunks:    text and table chunks with source/page/section/table metadata
- chunks_fts FTS5 (BM25, Porter stemming) over chunk text
- tables_catalog + tbl_<id>: every extracted table loaded as a SQL table, with
  parsed numeric (__num) and mean-separation letter (__letters) companion columns
- optional chunk embeddings (model2vec) for hybrid search, fused with BM25 by RRF
"""
import hashlib, json, re, sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY, source TEXT UNIQUE, md_path TEXT, sha256 TEXT,
  status TEXT, notes TEXT, added TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, doc_id INTEGER, kind TEXT, page INTEGER, section TEXT,
  table_id INTEGER, caption TEXT, text TEXT, emb BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, caption, section, content='chunks', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS tables_catalog(
  id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, caption TEXT,
  csv_path TEXT, sql_table TEXT, columns TEXT, n_rows INTEGER);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

# Plant-protection / regulatory synonyms used for query expansion (both directions).
SYNONYMS = [
    {"efficacy", "effectiveness", "efficacious", "performance"},
    {"crop safety", "phytotoxicity", "selectivity", "phytotoxic", "crop damage"},
    {"co-formulant", "coformulant", "formulant"},
    {"trial", "study", "experiment"},
    {"herbicide", "weed control"},
    {"fungicide", "disease control"},
    {"insecticide", "pest control"},
    {"rate", "dose", "dosage"},
    {"minor use", "minor crop"},
    {"zone", "zonal", "climatic zone"},
    {"resistance", "resistance management"},
    {"bridging", "comparability", "comparable"},
    {"reference product", "standard product", "reference"},
    {"untreated", "untreated control", "check"},
    {"gep", "good experimental practice"},
    {"bad", "biological assessment dossier"},
    {"drr", "draft registration report"},
]
STOP = set("""a an and are as at be by can do does for from has have how i in is it its of on or
should that the this to was what when where which who why will with within more than less
many much need needed needs required require there their them they about into any all some
we our you your me my""".split())

PAGE_RE = re.compile(r"<!-- page (\d+) -->")
TABLE_META_RE = re.compile(r"source: `([^`]+\.csv)`")
NUM_RE = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*([a-zA-Z]{1,6})?\s*$")


def connect(db, readonly=False):
    """Open the library. Shared/synced folders (e.g. a Windows folder mounted into a VM,
    network drives) often don't support SQLite file locking ('disk I/O error'); then
    fall back to no-lock mode with an in-memory journal, which is fine for one user."""
    db = Path(db).resolve()
    mode = "&mode=ro" if readonly else ""
    try:
        con = sqlite3.connect(f"file:{db}?{mode.lstrip('&')}" if readonly else str(db), uri=readonly)
        if not readonly:
            con.executescript(SCHEMA)
        else:
            con.execute("SELECT 1 FROM docs LIMIT 1")
        return con
    except sqlite3.OperationalError as e:
        if "disk I/O" not in str(e) and "locked" not in str(e):
            raise
    for j in (db.with_name(db.name + "-journal"),):  # stale journal from the failed attempt
        try:
            if j.exists() and j.stat().st_size <= 512:
                j.unlink()
        except OSError:
            pass
    con = sqlite3.connect(f"file:{db}?nolock=1{mode}", uri=True)
    if not readonly:
        con.execute("PRAGMA journal_mode=MEMORY")
        con.executescript(SCHEMA)
    return con


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- chunking

def dehyphenate(text):
    # 'biologi-\ncally' -> 'biologically'; keep short prefixes: 'co-\nformulant' -> 'co-formulant'
    def fix(m):
        left, right = m.group(1), m.group(2)
        return f"{left}-{right}" if len(left) <= 3 else f"{left}{right}"
    text = re.sub(r"\b(\w+)-\n(\w+)", fix, text)
    text = re.sub(r"[ \t]*\n(?!\n)[ \t]*", " ", text)   # join wrapped lines inside paragraphs
    return re.sub(r"[ \t]{2,}", " ", text)


def parse_front_matter(md):
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            return md[3:end], md[end + 4:]
    return "", md


def chunk_markdown(md, max_chars=1200, overlap=200):
    """Yield dicts: kind ('text'|'table'), page, section, caption, csv, text."""
    _, body = parse_front_matter(md)
    page, section = None, ""
    buf, out = [], []

    def flush():
        txt = dehyphenate("\n".join(buf)).strip()
        buf.clear()
        if len(txt) < 40:
            return
        start = 0
        while start < len(txt):
            end = min(len(txt), start + max_chars)
            if end < len(txt):  # break at a sentence end if possible
                cut = txt.rfind(". ", start + max_chars // 2, end)
                end = cut + 1 if cut != -1 else end
            out.append(dict(kind="text", page=page, section=section, caption="", csv=None,
                            text=txt[start:end].strip()))
            if end >= len(txt):
                break
            start = max(end - overlap, start + 1)

    lines = body.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = PAGE_RE.match(ln.strip())
        if m:
            flush(); page = int(m.group(1)); i += 1; continue
        if ln.startswith("#"):
            flush(); section = ln.lstrip("#").strip(); i += 1; continue
        if ln.startswith("|"):
            # caption + meta lines just above the table (already in buf)
            caption, csv = "", None
            tail = [b for b in buf[-4:] if b.strip()]
            for b in tail:
                mm = TABLE_META_RE.search(b)
                if mm:
                    csv = mm.group(1)
                elif b.startswith("**") or b.startswith("## Sheet"):
                    caption = b.strip("*# ").strip()
            for b in tail:  # remove caption/meta from text buffer
                if b in buf and (TABLE_META_RE.search(b) or b.startswith("**") or b.startswith("## Sheet")):
                    buf.remove(b)
            flush()
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append(lines[i]); i += 1
            out.append(dict(kind="table", page=page, section=section, caption=caption, csv=csv,
                            text=(caption + "\n" if caption else "") + "\n".join(tbl)))
            continue
        buf.append(ln); i += 1
    flush()
    return out


# ---------------------------------------------------------------- tables -> SQL

def _parsed_columns(df):
    """Add <col>__num and <col>__letters for columns that look like '82.5 b'."""
    for c in list(df.columns):
        vals = df[c].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) == 0:
            continue
        m = vals.str.extract(NUM_RE)
        if m[0].notna().mean() >= 0.6:
            mm = df[c].astype(str).str.extract(NUM_RE)
            df[f"{c}__num"] = pd.to_numeric(mm[0].str.replace(",", "."), errors="coerce")
            if mm[1].notna().any():
                df[f"{c}__letters"] = mm[1]
    return df


def load_table(con, doc_id, page, caption, csv_path, root=None):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() or f"col{i+1}" for i, c in enumerate(df.columns)]
    # make column names unique
    seen = {}
    cols = []
    for c in df.columns:
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = cols
    df = _parsed_columns(df)
    cur = con.execute("INSERT INTO tables_catalog(doc_id,page,caption,csv_path,columns,n_rows) "
                      "VALUES(?,?,?,?,?,?)", (doc_id, page, caption, _rel(csv_path, root),
                                              json.dumps(list(df.columns)), len(df)))
    tid = cur.lastrowid
    name = f"tbl_{tid}"
    df.to_sql(name, con, index=False, if_exists="replace")
    con.execute("UPDATE tables_catalog SET sql_table=? WHERE id=?", (name, tid))
    return tid


# ---------------------------------------------------------------- embeddings (optional)

_MODEL = None


def get_encoder(model_name="minishlab/potion-base-8M"):
    """model2vec static embeddings: numpy only, ~30 MB. Returns None if unavailable."""
    global _MODEL
    if _MODEL is None:
        try:
            from model2vec import StaticModel
            _MODEL = StaticModel.from_pretrained(model_name)
        except Exception:
            _MODEL = False
    return _MODEL or None


def embed(texts, encoder):
    v = np.asarray(encoder.encode(list(texts)), dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


# ---------------------------------------------------------------- indexing

def _rel(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())) if root else str(path)
    except ValueError:
        return str(path)


def remove_doc(con, source):
    row = con.execute("SELECT id FROM docs WHERE source=?", (source,)).fetchone()
    if not row:
        return
    did = row[0]
    for (cid, text, cap, sec) in con.execute(
            "SELECT id,text,caption,section FROM chunks WHERE doc_id=?", (did,)).fetchall():
        con.execute("INSERT INTO chunks_fts(chunks_fts,rowid,text,caption,section) VALUES('delete',?,?,?,?)",
                    (cid, text, cap, sec))
    con.execute("DELETE FROM chunks WHERE doc_id=?", (did,))
    for (t,) in con.execute("SELECT sql_table FROM tables_catalog WHERE doc_id=?", (did,)).fetchall():
        con.execute(f'DROP TABLE IF EXISTS "{t}"')
    con.execute("DELETE FROM tables_catalog WHERE doc_id=?", (did,))
    con.execute("DELETE FROM docs WHERE id=?", (did,))


def index_doc(con, source, md_path, sha, notes, encoder=None, root=None):
    remove_doc(con, source)
    md = Path(md_path).read_text(errors="replace")
    fm, _ = parse_front_matter(md)
    status = (re.search(r"^status:\s*(.+)$", fm, re.M) or [None, "unverified"])[1].split("#")[0].strip()
    did = con.execute("INSERT INTO docs(source,md_path,sha256,status,notes) VALUES(?,?,?,?,?)",
                      (source, _rel(md_path, root), sha, status, json.dumps(notes))).lastrowid
    chunks = chunk_markdown(md)
    ids = []
    tdir = Path(md_path).parent / (Path(md_path).stem + "_tables")
    nth = 0
    for ch in chunks:
        tid = None
        if ch["kind"] == "table":
            nth += 1
            if not ch["csv"] and (tdir / f"table_{nth:02d}.csv").exists():  # Word tables: by order
                ch["csv"] = str((tdir / f"table_{nth:02d}.csv").relative_to(Path(md_path).parent))
        if ch["kind"] == "table" and ch["csv"]:
            csv = Path(md_path).parent / ch["csv"]
            if csv.exists():
                tid = load_table(con, did, ch["page"], ch["caption"], csv, root)
        cid = con.execute("INSERT INTO chunks(doc_id,kind,page,section,table_id,caption,text) "
                          "VALUES(?,?,?,?,?,?,?)",
                          (did, ch["kind"], ch["page"], ch["section"], tid, ch["caption"], ch["text"])).lastrowid
        con.execute("INSERT INTO chunks_fts(rowid,text,caption,section) VALUES(?,?,?,?)",
                    (cid, ch["text"], ch["caption"], ch["section"]))
        ids.append(cid)
    if encoder is not None and chunks:
        vecs = embed([c["text"] for c in chunks], encoder)
        for cid, v in zip(ids, vecs):
            con.execute("UPDATE chunks SET emb=? WHERE id=?", (v.tobytes(), cid))
    return did, len(chunks)


# ---------------------------------------------------------------- search

def build_match(query):
    """Natural-language query -> FTS5 MATCH string with synonym expansion.
    Quoted phrases in the query are kept as phrases."""
    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]+"', " ", query).lower()
    terms = [t for t in re.findall(r"[\w][\w\-./()]*", rest) if t not in STOP and len(t) > 1]
    groups = []
    used = set()
    low = " " + rest + " "
    for syn in SYNONYMS:  # multi-word synonyms present in the query
        hit = [s for s in syn if f" {s} " in low or (" " not in s and s in terms)]
        if hit:
            groups.append(sorted(syn)); used.update(w for s in hit for w in s.split())
    for t in terms:
        if t not in used:
            groups.append([t])
    parts = [f'"{p}"' for p in phrases]
    for g in groups:
        alts = []
        for s in g:
            toks = [x for x in re.split(r"[^\w]+", s) if x]
            if toks:
                alts.append('"' + " ".join(toks) + '"')
        if alts:
            parts.append("(" + " OR ".join(alts) + ")" if len(alts) > 1 else alts[0])
    return parts


def match_strings(query):
    """(strict, loose): all concepts required vs any concept."""
    parts = build_match(query)
    if not parts:
        return "", ""
    return " AND ".join(parts), " OR ".join(parts)


def search(con, query, k=8, kind=None, source=None, encoder=None):
    where, args = [], []
    if kind:
        where.append("c.kind=?"); args.append(kind)
    if source:
        where.append("d.source LIKE ?"); args.append(f"%{source}%")
    w = (" AND " + " AND ".join(where)) if where else ""
    strict, loose = match_strings(query)
    match = strict
    fused = {}

    def bm25(m):
        return con.execute(f"""
            SELECT c.id, bm25(chunks_fts, 1.0, 2.0, 0.5) AS s FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid JOIN docs d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ? {w} ORDER BY s LIMIT ?""", [m, *args, max(50, k * 5)]).fetchall()

    if strict:
        # chunks containing every concept rank above chunks matching only some
        # a table caption summarises the whole table, so caption hits get extra weight
        runs = ((2.0, strict), (1.0, loose), (1.5, "{caption} : (" + strict + ")"),
                (0.5, "{caption} : (" + loose + ")"))
        for weight, m in runs:
            for r, (cid, _) in enumerate(bm25(m)):
                fused[cid] = fused.get(cid, 0) + weight / (60 + r)
    mode = "keyword"
    has_emb = con.execute("SELECT 1 FROM chunks WHERE emb IS NOT NULL LIMIT 1").fetchone()
    if encoder is not None and has_emb:
        rows = con.execute(f"""SELECT c.id, c.emb FROM chunks c JOIN docs d ON d.id=c.doc_id
                               WHERE c.emb IS NOT NULL {w}""", args).fetchall()
        if rows:
            q = embed([query], encoder)[0]
            M = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            order = np.argsort(-(M @ q))[: max(50, k * 5)]
            for r, j in enumerate(order):
                cid = rows[j][0]
                fused[cid] = fused.get(cid, 0) + 1.0 / (60 + r)
            mode = "hybrid"
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    out = []
    for cid in top:
        r = con.execute("""SELECT c.id, d.source, d.status, c.kind, c.page, c.section, c.caption,
                           c.table_id, t.sql_table, t.csv_path, c.text
                           FROM chunks c JOIN docs d ON d.id=c.doc_id
                           LEFT JOIN tables_catalog t ON t.id=c.table_id WHERE c.id=?""", (cid,)).fetchone()
        out.append(dict(zip(["chunk", "source", "status", "kind", "page", "section", "caption",
                             "table_id", "sql_table", "csv", "text"], r), score=round(fused[cid], 4)))
    return out, mode, match


def write_index_md(con, outdir):
    rows = con.execute("""SELECT d.source, d.md_path, d.status, d.notes,
                          (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id),
                          (SELECT COUNT(*) FROM tables_catalog t WHERE t.doc_id=d.id)
                          FROM docs d ORDER BY d.source""").fetchall()
    lines = ["# Library index\n", "| Source | Markdown | Chunks | Tables | Status | Notes |",
             "|---|---|---|---|---|---|"]
    for s, md, st, notes, nc, nt in rows:
        try:
            rel = Path(md).relative_to(outdir)
        except ValueError:
            rel = md
        n = "; ".join(json.loads(notes or "[]")) or "-"
        lines.append(f"| {Path(s).name} | `{rel}` | {nc} | {nt} | {st} | {n} |")
    (Path(outdir) / "INDEX.md").write_text("\n".join(lines) + "\n")
