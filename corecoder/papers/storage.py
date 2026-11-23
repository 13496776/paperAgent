"""SQLite paper/chunk records with recoverable Chroma vector indexing."""
import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from .corpus import DEFAULT_EMBED_MODEL, chunks_for_paper
from .retrieval import passage_text, tokenize


@lru_cache(maxsize=8)
def _client(path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def _collection_name(model):
    return "papers_" + hashlib.sha256(model.encode()).hexdigest()[:24]


def _vector_id(owner, paper_id, chunk_id):
    return hashlib.sha256(json.dumps([owner, paper_id, chunk_id], ensure_ascii=False).encode()).hexdigest()


class CorpusStore:
    def __init__(self, path, chroma_path=None):
        self.path = str(path)
        default = str(Path(self.path).with_suffix("")) + "-chroma"
        self.chroma_path = str(Path(chroma_path or os.environ.get("PAPER_CHROMA_PATH") or default).resolve())
        self.model = os.environ.get("PAPER_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS papers (
                    owner TEXT NOT NULL, id TEXT NOT NULL, title TEXT NOT NULL, authors TEXT NOT NULL,
                    source TEXT NOT NULL, doi TEXT NOT NULL, ccf_rank TEXT, page_count INTEGER,
                    status TEXT NOT NULL, model TEXT, error TEXT, created_at REAL NOT NULL, generation TEXT NOT NULL,
                    chunk_size INTEGER, chunk_overlap INTEGER, index_version TEXT NOT NULL,
                    venue TEXT NOT NULL, year INTEGER, ccf_source TEXT NOT NULL, ccf_version TEXT NOT NULL,
                    ccf_verified INTEGER NOT NULL, updated_at REAL NOT NULL, vector_path TEXT,
                    PRIMARY KEY(owner, id)
                );
                CREATE TABLE IF NOT EXISTS paper_chunks (
                    owner TEXT NOT NULL, paper_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, section TEXT NOT NULL, page_start INTEGER, page_end INTEGER,
                    text TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(owner, paper_id, chunk_id),
                    FOREIGN KEY(owner, paper_id) REFERENCES papers(owner, id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS paper_chunks_owner ON paper_chunks(owner, paper_id);
                CREATE TABLE IF NOT EXISTS paper_vector_collections (
                    owner TEXT NOT NULL, paper_id TEXT NOT NULL, collection TEXT NOT NULL, path TEXT NOT NULL,
                    ever_ready INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(owner, paper_id, collection, path),
                    FOREIGN KEY(owner, paper_id) REFERENCES papers(owner, id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS legacy_paper_migration (
                    owner TEXT NOT NULL, id TEXT NOT NULL, PRIMARY KEY(owner, id)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunk_fts USING fts5(
                    owner UNINDEXED, paper_id UNINDEXED, chunk_id UNINDEXED, tokens
                );
            ''')
            self._migrate_legacy(db)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def _collection(self, name, path=None):
        return _client(path or self.chroma_path).get_or_create_collection(
            name=name, embedding_function=None, metadata={"hnsw:space": "cosine"})

    def _insert(self, db, owner, paper, chunks):
        authors = json.dumps(paper.get("authors", ""), ensure_ascii=False)
        db.execute("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'indexing', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                   (owner, paper["id"], paper["title"], authors, paper.get("source") or "",
                    paper.get("doi") or "", paper.get("ccf_rank"), paper.get("page_count"), time.time(), uuid.uuid4().hex,
                    paper.get("chunk_size"), paper.get("chunk_overlap"), paper.get("index_version", "legacy"),
                    paper.get("venue") or "", paper.get("year"), paper.get("ccf_source") or "",
                    paper.get("ccf_version") or "", int(bool(paper.get("ccf_verified"))), time.time()))
        used = set()
        for ordinal, item in enumerate(chunks):
            chunk = dict(item)
            ident = chunk.get("id") or hashlib.sha256(f"{paper['id']}:{ordinal}".encode()).hexdigest()[:20]
            if ident in used:
                ident = hashlib.sha256(f"{ident}:{ordinal}".encode()).hexdigest()[:20]
            used.add(ident)
            chunk.update(id=ident, paper_id=paper["id"])
            db.execute("INSERT INTO paper_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (owner, paper["id"], ident, ordinal, chunk.get("section") or "", chunk.get("page_start"),
                        chunk.get("page_end"), chunk["text"], json.dumps(chunk, ensure_ascii=False)))
            db.execute("INSERT INTO paper_chunk_fts(owner,paper_id,chunk_id,tokens) VALUES (?, ?, ?, ?)",
                       (owner, paper["id"], ident, " ".join(tokenize(passage_text(chunk)))))

    def _migrate_legacy(self, db):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_corpus'").fetchone()
        if not exists:
            return
        missing = db.execute("SELECT 1 FROM paper_corpus WHERE NOT EXISTS "
                             "(SELECT 1 FROM legacy_paper_migration m WHERE m.owner=paper_corpus.owner "
                             "AND m.id=paper_corpus.id) LIMIT 1").fetchone()
        if not missing:
            return
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT owner,id,metadata,chunks FROM paper_corpus WHERE NOT EXISTS "
                          "(SELECT 1 FROM legacy_paper_migration m WHERE m.owner=paper_corpus.owner "
                          "AND m.id=paper_corpus.id)").fetchall()
        for owner, ident, metadata, raw_chunks in rows:
            paper = json.loads(metadata)
            paper["id"] = ident
            chunks = json.loads(raw_chunks)
            if not db.execute("SELECT 1 FROM papers WHERE owner=? AND id=?", (owner, ident)).fetchone():
                self._insert(db, owner, paper, chunks)
            db.execute("INSERT INTO legacy_paper_migration VALUES (?, ?)", (owner, ident))

    def add(self, owner, paper, encoder):
        rank = paper.get("ccf_rank")
        if rank not in (None, "A", "B", "C", "无类别"):
            raise ValueError("CCF rank must be A, B, C, 无类别 or empty")
        year = paper.get("year")
        if year is not None and (isinstance(year, bool) or not isinstance(year, int) or not 1000 <= year <= 9999):
            raise ValueError("year must be a four-digit integer")
        if paper.get("ccf_verified") and (not rank or not paper.get("ccf_source") or not paper.get("ccf_version")):
            raise ValueError("verified CCF rank requires rank, source and version")
        with self._connect() as db:
            existing = db.execute("SELECT status FROM papers WHERE owner=? AND id=?", (owner, paper["id"])).fetchone()
        created = existing is None
        if created:
            paper = dict(paper)
            paper["chunk_size"] = int(os.environ.get("PAPER_CHUNK_SIZE", "1024"))
            paper["chunk_overlap"] = int(os.environ.get("PAPER_CHUNK_OVERLAP", "50"))
            paper["index_version"] = "semantic-v1"
            chunks = chunks_for_paper(paper, embedding_model=encoder)
            if not chunks:
                raise ValueError("paper contains no searchable text")
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if not db.execute("SELECT 1 FROM papers WHERE owner=? AND id=?", (owner, paper["id"])).fetchone():
                    self._insert(db, owner, paper, chunks)
                else:
                    created = False
        self.ensure_indexed(owner, encoder, paper_id=paper["id"])
        result = next((item for item in self.papers(owner) if item["id"] == paper["id"]), None)
        if result is None:
            raise ValueError("paper was deleted while indexing")
        return {**result, "chunks": len(self.chunks(owner, paper["id"])), "created": created}

    def ensure_indexed(self, owner, encoder, paper_id=None, progress=None):
        with self._connect() as db:
            query = "SELECT id,status,model,vector_path FROM papers WHERE owner=?"
            args = [owner]
            if paper_id is not None:
                query += " AND id=?"
                args.append(paper_id)
            rows = db.execute(query, args).fetchall()
        for number, (ident, status, model, vector_path) in enumerate(rows, 1):
            if status == "deleting":
                self.delete(owner, ident)
            elif status != "ready" or model != self.model or vector_path != self.chroma_path or not Path(self.chroma_path).exists():
                if progress:
                    progress("index_build", f"正在构建或恢复论文向量索引（{number}/{len(rows)}）")
                self._index(owner, ident, encoder)

    def _index(self, owner, ident, encoder):
        name = _collection_name(self.model)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,model,generation,vector_path FROM papers WHERE owner=? AND id=?", (owner, ident)).fetchone()
            if not row or row[0] == "deleting" or (row[0] == "ready" and row[1] == self.model and row[3] == self.chroma_path
                                                  and Path(self.chroma_path).exists()):
                return
            generation = row[2]
            db.execute("UPDATE papers SET status='indexing',error=NULL,updated_at=? WHERE owner=? AND id=?", (time.time(), owner, ident))
            db.execute("INSERT OR IGNORE INTO paper_vector_collections VALUES (?, ?, ?, ?, 0)", (owner, ident, name, self.chroma_path))
        chunks = self.chunks(owner, ident, include_pending=True)
        try:
            if not chunks:
                raise ValueError("paper has no chunks to index")
            vectors = encoder.encode([passage_text(c) for c in chunks], normalize_embeddings=True)
            embeddings = [[float(x) for x in vector] for vector in vectors]
            if len(embeddings) != len(chunks):
                raise ValueError("embedding count does not match chunks")
            ids = [_vector_id(owner, ident, c["id"]) for c in chunks]
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT status,generation FROM papers WHERE owner=? AND id=?", (owner, ident)).fetchone()
                if not row or row[0] == "deleting" or row[1] != generation:
                    return
                collection = self._collection(name)
                batch = _client(self.chroma_path).get_max_batch_size()
                for start in range(0, len(ids), batch):
                    end = start + batch
                    collection.upsert(ids=ids[start:end], embeddings=embeddings[start:end],
                                      metadatas=[{"owner": owner, "paper_id": ident, "chunk_id": c["id"]}
                                                 for c in chunks[start:end]])
                    if len(collection.get(ids=ids[start:end], include=[])["ids"]) != len(ids[start:end]):
                        raise RuntimeError("vector count does not match chunk count")
                db.execute("UPDATE paper_vector_collections SET ever_ready=1 WHERE owner=? AND paper_id=? AND collection=? AND path=?",
                           (owner, ident, name, self.chroma_path))
                db.execute("UPDATE papers SET status='ready',model=?,error=NULL,updated_at=?,vector_path=? WHERE owner=? AND id=?",
                           (self.model, time.time(), self.chroma_path, owner, ident))
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE papers SET status='indexing',error=?,updated_at=? WHERE owner=? AND id=? AND status='indexing' AND generation=?",
                           (str(exc)[:1000], time.time(), owner, ident, generation))
            raise

    def papers(self, owner):
        with self._connect() as db:
            pending = [row[0] for row in db.execute("SELECT id FROM papers WHERE owner=? AND status='deleting'", (owner,))]
        for ident in pending:
            self.delete(owner, ident)
        with self._connect() as db:
            rows = db.execute("SELECT id,title,authors,source,doi,ccf_rank,page_count,status,error,chunk_size,chunk_overlap,index_version,model,venue,year,ccf_source,ccf_version,ccf_verified,created_at,updated_at FROM papers "
                              "WHERE owner=? AND status!='deleting' ORDER BY created_at DESC", (owner,)).fetchall()
        return [{"id": r[0], "title": r[1], "authors": json.loads(r[2]), "source": r[3], "doi": r[4],
                 "ccf_rank": r[5], "page_count": r[6], "status": r[7], "error": r[8], "chunk_size": r[9],
                 "chunk_overlap": r[10], "index_version": r[11], "embedding_model": r[12], "venue": r[13], "year": r[14],
                 "ccf_source": r[15], "ccf_version": r[16], "ccf_verified": bool(r[17]),
                 "created_at": r[18], "updated_at": r[19]} for r in rows]

    def chunks(self, owner, paper_id=None, include_pending=False):
        with self._connect() as db:
            query = "SELECT c.payload FROM paper_chunks c JOIN papers p ON p.owner=c.owner AND p.id=c.paper_id WHERE c.owner=?"
            args = [owner]
            if paper_id is not None:
                query += " AND c.paper_id=?"
                args.append(paper_id)
            if not include_pending:
                query += " AND p.status='ready'"
            query += " ORDER BY c.paper_id,c.ordinal"
            return [json.loads(row[0]) for row in db.execute(query, args)]

    def chunks_for_keys(self, owner, keys):
        found = {}
        with self._connect() as db:
            for start in range(0, len(keys), 200):
                batch = keys[start:start + 200]
                conditions = " OR ".join("(c.paper_id=? AND c.chunk_id=?)" for _ in batch)
                params = [owner]
                for paper_id, chunk_id in batch:
                    params.extend((paper_id, chunk_id))
                rows = db.execute("SELECT c.paper_id,c.chunk_id,c.payload FROM paper_chunks c JOIN papers p "
                                  "ON p.owner=c.owner AND p.id=c.paper_id WHERE c.owner=? AND p.status='ready' AND ("
                                  + conditions + ")", params).fetchall()
                found.update({(row[0], row[1]): json.loads(row[2]) for row in rows})
        return found

    def lexical_search(self, owner, query, limit):
        terms = list(dict.fromkeys(tokenize(query)))[:64]
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self._connect() as db:
            rows = db.execute("SELECT f.paper_id,f.chunk_id FROM paper_chunk_fts f JOIN papers p "
                              "ON p.owner=f.owner AND p.id=f.paper_id WHERE paper_chunk_fts MATCH ? "
                              "AND f.owner=? AND p.status='ready' ORDER BY bm25(paper_chunk_fts) LIMIT ?",
                              (expression, owner, limit)).fetchall()
        return rows

    def vector_search(self, owner, query_vector, limit):
        with self._connect() as db:
            ready = [row[0] for row in db.execute("SELECT id FROM papers WHERE owner=? AND status='ready' AND model=? AND vector_path=?",
                                                  (owner, self.model, self.chroma_path))]
        if not ready:
            return []
        collection = self._collection(_collection_name(self.model))
        result = collection.query(query_embeddings=[[float(v) for v in query_vector]], n_results=limit,
                                  where={"$and": [{"owner": owner}, {"paper_id": {"$in": ready}}]},
                                  include=["metadatas"])
        return [(m["paper_id"], m["chunk_id"]) for m in result["metadatas"][0]]

    def shared_vector_search(self, chunks, query_vector, encoder, limit, progress=None):
        fingerprint = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        collection = self._collection(_collection_name("shared:" + self.model + ":" + fingerprint))
        ids = [str(i) for i in range(len(chunks))]
        batch_size = _client(self.chroma_path).get_max_batch_size()
        if collection.count() != len(ids):
            if progress:
                progress("index_build", f"构建或恢复共享语料索引，检查 {len(ids)} 个分块")
            for start in range(0, len(ids), batch_size):
                expected = ids[start:start + batch_size]
                existing = set(collection.get(ids=expected, include=[])["ids"])
                missing = [ident for ident in expected if ident not in existing]
                if not missing:
                    continue
                vectors = encoder.encode([passage_text(chunks[int(ident)]) for ident in missing], normalize_embeddings=True)
                embeddings = [[float(value) for value in vector] for vector in vectors]
                if len(embeddings) != len(missing):
                    raise ValueError("embedding count does not match shared chunks")
                collection.upsert(ids=missing, embeddings=embeddings)
            if collection.count() != len(ids):
                raise RuntimeError("shared vector count does not match chunk count")
        result = collection.query(query_embeddings=[[float(value) for value in query_vector]],
                                  n_results=min(limit, len(ids)), include=[])
        return [int(ident) for ident in result["ids"][0]]

    def delete(self, owner, ident):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM papers WHERE owner=? AND id=?", (owner, ident)).fetchone()
            if not row:
                return False
            db.execute("UPDATE papers SET status='deleting',updated_at=? WHERE owner=? AND id=?", (time.time(), owner, ident))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            locations = db.execute("SELECT collection,path,ever_ready FROM paper_vector_collections WHERE owner=? AND paper_id=?",
                                   (owner, ident)).fetchall()
            for name, path, ever_ready in locations:
                if not Path(path).exists():
                    if ever_ready:
                        raise RuntimeError(f"Vector index directory is unavailable: {path}")
                    continue
                self._collection(name, path).delete(where={"$and": [{"owner": owner}, {"paper_id": ident}]})
            db.execute("DELETE FROM paper_chunk_fts WHERE owner=? AND paper_id=?", (owner, ident))
            db.execute("DELETE FROM paper_chunks WHERE owner=? AND paper_id=?", (owner, ident))
            db.execute("DELETE FROM paper_vector_collections WHERE owner=? AND paper_id=?", (owner, ident))
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_corpus'").fetchone():
                db.execute("DELETE FROM paper_corpus WHERE owner=? AND id=?", (owner, ident))
            db.execute("DELETE FROM papers WHERE owner=? AND id=?", (owner, ident))
        return True
