"""Storage: SQLite for the `trips` table plus a tiny document store (collections of JSON docs).

The doc store mirrors the plan's MongoDB collections (daily_metrics, alerts, agent_runs,
eval_results, state) on top of SQLite so the app runs with zero external services. The
function signatures are deliberately Mongo-shaped, so swapping in pymongo/Atlas later
only touches this file.
"""
import json
import sqlite3
from contextlib import contextmanager

from .config import DATA, DB_PATH


@contextmanager
def conn():
    DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute(
        "CREATE TABLE IF NOT EXISTS docs (collection TEXT NOT NULL, id INTEGER NOT NULL, "
        "doc TEXT NOT NULL, PRIMARY KEY (collection, id))"
    )
    try:
        yield c
        c.commit()
    finally:
        c.close()


def insert(collection: str, doc: dict) -> int:
    with conn() as c:
        nid = c.execute("SELECT COALESCE(MAX(id),0)+1 FROM docs WHERE collection=?", (collection,)).fetchone()[0]
        doc = {**doc, "id": nid}
        c.execute("INSERT INTO docs VALUES (?,?,?)", (collection, nid, json.dumps(doc)))
    return nid


def put(collection: str, id_: int, doc: dict):
    doc = {**doc, "id": id_}
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?)", (collection, id_, json.dumps(doc)))


def get(collection: str, id_: int):
    with conn() as c:
        row = c.execute("SELECT doc FROM docs WHERE collection=? AND id=?", (collection, id_)).fetchone()
    return json.loads(row[0]) if row else None


def all_docs(collection: str) -> list:
    with conn() as c:
        rows = c.execute("SELECT doc FROM docs WHERE collection=? ORDER BY id", (collection,)).fetchall()
    return [json.loads(r[0]) for r in rows]


def clear(collection: str):
    with conn() as c:
        c.execute("DELETE FROM docs WHERE collection=?", (collection,))


def drop_trips():
    with conn() as c:
        c.execute("DROP TABLE IF EXISTS trips")
