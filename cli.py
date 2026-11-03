"""dossier-rag command line.

  dossier-rag ingest  <files|folders|.zip> --lib library [--semantic]
  dossier-rag search  "query" --lib library [-k 8] [--kind text|table] [--source NAME] [--full] [--json]
  dossier-rag tables  --lib library [--find WORD]
  dossier-rag sql     "SELECT ..." --lib library
  dossier-rag show    CHUNK_ID --lib library
  dossier-rag reindex SOURCE --lib library     (after hand-fixing its .md / CSVs)
  dossier-rag verify  SOURCE --lib library --note "what was checked"
  dossier-rag stats   --lib library
  dossier-rag pack    --lib library --out library.zip
  dossier-rag restore library.zip --lib library
"""
import argparse, json, sys
from pathlib import Path

from . import store
from .convert import convert_all


def _con(lib):
    lib = Path(lib)
    lib.mkdir(parents=True, exist_ok=True)
    return store.connect(lib / "dossier.db"), lib


def _is_library_zip(p):
    import zipfile
    p = Path(p)
    if p.suffix.lower() != ".zip" or not p.is_file():
        return False
    with zipfile.ZipFile(p) as z:
        return any(Path(n).name == "dossier.db" for n in z.namelist())


def restore(zip_path, lib):
    """Unpack a packed library into lib (merging is not supported: lib must be new or empty)."""
    import shutil, tempfile
    from .convert import safe_extract
    lib = Path(lib)
    if (lib / "dossier.db").exists():
        raise RuntimeError(f"{lib}/dossier.db already exists; restore into a new --lib folder")
    tmp = Path(tempfile.mkdtemp(prefix="dossier_restore_"))
    safe_extract(zip_path, tmp)
    db = next(tmp.rglob("dossier.db"))
    lib.mkdir(parents=True, exist_ok=True)
    for item in db.parent.iterdir():
        shutil.move(str(item), str(lib / item.name))
    shutil.rmtree(tmp, ignore_errors=True)


def cmd_restore(a):
    restore(a.zip, a.lib)
    con, lib = _con(a.lib)
    print(f"Library restored to {lib}.")
    return cmd_stats(a)


def cmd_pack(a):
    import zipfile
    lib = Path(a.lib)
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(lib.rglob("*")):
            rel = f.relative_to(lib)
            if f.is_file() and "_extracted" not in rel.parts:
                z.write(f, Path("library") / rel)
    print(f"Packed {lib} -> {out} ({out.stat().st_size // 1024} KB). Upload this zip in a new chat to continue.")
    return 0


def cmd_ingest(a):
    inputs = []
    for i in a.inputs:  # a previously packed library is restored, not converted
        if _is_library_zip(i):
            if (Path(a.lib) / "dossier.db").exists():
                print(f"SKIP {i}: looks like a packed library, but {a.lib} already has one")
            else:
                restore(i, a.lib)
                print(f"RESTORED library from {i}")
        else:
            inputs.append(i)
    con, lib = _con(a.lib)
    if not inputs:
        return cmd_stats(a)
    results, skipped = convert_all(inputs, lib / "docs")
    enc = None
    if a.semantic:
        enc = store.get_encoder()
        print("semantic: " + ("model loaded" if enc else "model not available here; keyword search only"))
        if enc:
            con.execute("INSERT OR REPLACE INTO meta VALUES('semantic','1')")
    for r in results:
        if not r["md"]:
            continue
        # source key = path inside the library (stable across re-ingests of the same file)
        src = str(Path(r["md"]).relative_to(lib / "docs").with_suffix("")) + Path(r["src"]).suffix
        did, n = store.index_doc(con, src, r["md"], store.sha256(r["src"]), r["notes"], enc, root=lib)
        print(f"     indexed {n} chunks")
    con.commit()
    store.write_index_md(con, lib)
    ok = sum(1 for r in results if r["md"])
    print(f"\n{ok} converted+indexed, {len(results) - ok} failed, {len(skipped)} skipped. "
          f"Library: {lib}/dossier.db, index: {lib}/INDEX.md")
    return 0 if ok or not results else 1


def cmd_search(a):
    con, lib = _con(a.lib)
    enc = None
    if con.execute("SELECT value FROM meta WHERE key='semantic'").fetchone():
        enc = store.get_encoder()
    hits, mode, match = store.search(con, a.query, a.k, a.kind, a.source, enc)
    if a.json:
        print(json.dumps(dict(mode=mode, match=match, hits=hits), indent=1, ensure_ascii=False))
        return 0
    print(f"[{mode} search] {len(hits)} hits   (FTS query: {match})\n")
    for h in hits:
        cite = f"{h['source']}, p.{h['page']}" if h["page"] else h["source"]
        tag = f"TABLE {h['sql_table']} " if h["kind"] == "table" else ""
        head = f"#{h['chunk']} {tag}[{cite}]"
        if h["caption"]:
            head += f"  {h['caption'][:90]}"
        if h["status"] and not h["status"].startswith("verified"):
            head += "  (unverified)"
        print(head)
        txt = h["text"] if a.full else (h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))
        print("   " + txt.replace("\n", "\n   ") + "\n")
    return 0


def cmd_tables(a):
    con, _ = _con(a.lib)
    q = """SELECT t.sql_table, d.source, t.page, t.n_rows, t.caption, t.columns
           FROM tables_catalog t JOIN docs d ON d.id=t.doc_id"""
    args = []
    if a.find:
        q += " WHERE t.caption LIKE ? OR t.columns LIKE ? OR d.source LIKE ?"
        args = [f"%{a.find}%"] * 3
    for name, src, page, n, cap, cols in con.execute(q + " ORDER BY t.id", args):
        print(f"{name}  [{src}{', p.' + str(page) if page else ''}]  {n} rows  {cap or ''}")
        print("    columns: " + ", ".join(json.loads(cols)))
    return 0


def cmd_sql(a):
    import sqlite3
    db = (Path(a.lib) / "dossier.db").resolve()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # read-only
    if not a.query.lstrip().lower().startswith(("select", "with", "pragma")):
        print("Only read-only SELECT/WITH queries are allowed.", file=sys.stderr)
        return 2
    cur = con.execute(a.query)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    import pandas as pd
    print(pd.DataFrame(rows, columns=cols).to_markdown(index=False, disable_numparse=True) if rows else "(no rows)")
    return 0


def cmd_show(a):
    con, _ = _con(a.lib)
    r = con.execute("""SELECT d.source, c.page, c.kind, c.caption, c.text, t.csv_path
                       FROM chunks c JOIN docs d ON d.id=c.doc_id
                       LEFT JOIN tables_catalog t ON t.id=c.table_id WHERE c.id=?""", (a.chunk,)).fetchone()
    if not r:
        print("no such chunk"); return 1
    r = list(r)
    if r[5]:
        r[5] = str(Path(a.lib) / r[5])
    print(f"[{r[0]}, p.{r[1]}] {r[2]} {r[3] or ''}\n" + (f"csv: {r[5]}\n" if r[5] else "") + "\n" + r[4])
    return 0


def cmd_reindex(a):
    """Re-index from the (hand-corrected) Markdown + CSVs without converting again."""
    con, lib = _con(a.lib)
    rows = con.execute("SELECT source, md_path, sha256, notes, status FROM docs WHERE source LIKE ?",
                       (f"%{a.source}%",)).fetchall()
    enc = store.get_encoder() if con.execute("SELECT value FROM meta WHERE key='semantic'").fetchone() else None
    for src, md, sha, notes, status in rows:
        did, n = store.index_doc(con, src, lib / md, sha, json.loads(notes or "[]"), enc, root=lib)
        con.execute("UPDATE docs SET status=? WHERE id=?", (status, did))
        print(f"reindexed {src}: {n} chunks")
    con.commit()
    store.write_index_md(con, lib)
    return 0 if rows else 1


def cmd_verify(a):
    con, _ = _con(a.lib)
    n = con.execute("UPDATE docs SET status=? WHERE source LIKE ?",
                    (f"verified ({a.note})", f"%{a.source}%")).rowcount
    con.commit()
    print(f"{n} document(s) marked verified")
    return 0 if n else 1


def cmd_stats(a):
    con, lib = _con(a.lib)
    d = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    c = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    t = con.execute("SELECT COUNT(*) FROM tables_catalog").fetchone()[0]
    e = con.execute("SELECT COUNT(*) FROM chunks WHERE emb IS NOT NULL").fetchone()[0]
    print(f"{d} documents, {c} chunks ({e} with embeddings), {t} SQL tables. See {lib}/INDEX.md")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dossier-rag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, **kw):
        s = sub.add_parser(name, **kw)
        s.add_argument("--lib", default="library", help="library folder (default: ./library)")
        s.set_defaults(fn=fn)
        return s

    s = p("ingest", cmd_ingest, help="convert + index files, folders, .zip")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--semantic", action="store_true", help="also compute embeddings if a model is available")
    s = p("search", cmd_search, help="keyword (+semantic) search with citations")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--kind", choices=["text", "table"])
    s.add_argument("--source")
    s.add_argument("--full", action="store_true")
    s.add_argument("--json", action="store_true")
    s = p("tables", cmd_tables, help="list extracted tables and their SQL names")
    s.add_argument("--find")
    s = p("sql", cmd_sql, help="read-only SQL over the tables")
    s.add_argument("query")
    s = p("show", cmd_show, help="print one chunk in full")
    s.add_argument("chunk", type=int)
    s = p("reindex", cmd_reindex, help="re-index a document from its edited Markdown/CSVs")
    s.add_argument("source")
    s = p("verify", cmd_verify, help="mark a document as checked against the original")
    s.add_argument("source")
    s.add_argument("--note", required=True)
    p("stats", cmd_stats, help="library summary")
    s = p("pack", cmd_pack, help="zip the library to carry it to another chat")
    s.add_argument("--out", default="library.zip")
    s = p("restore", cmd_restore, help="unpack a library zip made by 'pack'")
    s.add_argument("zip")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
