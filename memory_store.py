import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from models import ollama_ef

MEMORY_JSON_PATH = "long_term_memory.json"
MEMORY_VECTOR_DIR = "./memory_vector_db"

class LongTermMemory:
    def __init__(
        self,
        json_path: str = MEMORY_JSON_PATH,
        vector_dir: str = MEMORY_VECTOR_DIR,
    ):
        self.json_path = json_path

        client = chromadb.PersistentClient(path=vector_dir)
        self.collection = client.get_or_create_collection(
            name="agent_lessons",
            embedding_function=ollama_ef,
        )

        self._ensure_json_file()

    def _ensure_json_file(self):
        if not os.path.exists(self.json_path) or os.path.getsize(self.json_path) == 0:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] {self.json_path} contains invalid JSON. Defaulting to empty memory.")
            return []

    def _save(self, data: List[Dict[str, Any]]):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _next_id(self, data: List[Dict[str, Any]]) -> str:
        return f"fb{len(data) + 1}"

    def add_lesson(
        self,
        lesson: str,
        task_description: str,
        chat_id: Optional[str] = None,
        had_correction: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = self._load()
        entry_id = self._next_id(data)
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": entry_id,
            "lesson": lesson,
            "chat_id": chat_id,
            "timestamp": timestamp,
        }
        data.append(entry)
        self._save(data)

        doc_metadata = {
            "chat_id": chat_id or "unknown",
            "timestamp": timestamp,
            **(metadata or {}),
        }

        self.collection.add(
            ids=[entry_id],
            documents=[lesson],
            metadatas=[doc_metadata],
        )

        return entry_id

    def recall(self, query: str, k: int = 3) -> str:
        """Return the k most relevant past lessons, formatted for prompt injection."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
            )
        except Exception as e:
            print(f"[WARN] Memory recall failed: {e}")
            return ""

        documents = results.get("documents")
        if not documents or not documents[0]:
            return ""

        return "\n".join(f"- {doc}" for doc in documents[0])