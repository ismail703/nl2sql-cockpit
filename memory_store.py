import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from models import get_embedding, client

MEMORY_JSON_PATH = "agent_memory.json"
MEMORY_VECTOR_DIR = "./memory_vector_db"


class LongTermMemory:
    def __init__(
        self,
        json_path: str = MEMORY_JSON_PATH,
        vector_dir: str = MEMORY_VECTOR_DIR,
    ):
        self.json_path = json_path
        self.collection_name = "agent_lessons"

        if not client.collection_exists(self.collection_name):
            dummy_vector = get_embedding("test")

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(dummy_vector),
                    distance=models.Distance.COSINE,
                ),
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
            print(
                f"[WARN] {self.json_path} contains invalid JSON. Defaulting to empty memory."
            )
            return []

    def _save(self, data: List[Dict[str, Any]]):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _next_id_int(self, data: List[Dict[str, Any]]) -> int:
        return len(data) + 1

    def add_lesson(
        self,
        lesson: str,
        chat_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = self._load()

        entry_id_int = self._next_id_int(data)
        entry_id_str = str(entry_id_int)

        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": entry_id_str,
            "lesson": lesson,
            "chat_id": chat_id,
            "timestamp": timestamp,
        }

        data.append(entry)
        self._save(data)

        payload = {
            "document": lesson,
            "chat_id": chat_id or "unknown",
            "timestamp": timestamp,
            **(metadata or {}),
        }

        vector = get_embedding(lesson)

        client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=entry_id_int,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        return entry_id_str

    def recall(self, query: str, k: int = 3) -> str:
        try:
            query_vector = get_embedding(query)

            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
            )

        except Exception as e:
            print(f"[WARN] Memory recall failed: {e}")
            return ""

        if not response.points:
            return ""

        lessons = []

        for point in response.points:
            if point.payload and "document" in point.payload:
                lessons.append(point.payload["document"])

        return "\n".join(f"- {lesson}" for lesson in lessons)

    def recall_with_ids(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Like recall(), but returns structured results with IDs so callers
        can update/delete specific entries. score_threshold filters out
        weak matches (Qdrant cosine similarity, higher = more similar).
        """
        try:
            query_vector = get_embedding(query)

            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                score_threshold=score_threshold,
            )
        except Exception as e:
            print(f"[WARN] Memory recall_with_ids failed: {e}")
            return []

        results = []
        for point in response.points:
            if point.payload and "document" in point.payload:
                results.append({
                    "id": str(point.id),
                    "lesson": point.payload["document"],
                    "score": point.score,
                })

        return results

    def update_lesson(self, entry_id: str, lesson: str) -> bool:
        """
        Overwrite the lesson text for entry_id in both the JSON store
        and Qdrant (re-embeds the new text, keeps existing chat_id/metadata).
        """
        data = self._load()
        entry = next((e for e in data if e["id"] == str(entry_id)), None)

        if entry is None:
            print(f"[WARN] update_lesson: no entry with id {entry_id}")
            return False

        timestamp = datetime.now(timezone.utc).isoformat()
        entry["lesson"] = lesson
        entry["timestamp"] = timestamp
        self._save(data)

        try:
            point_id = int(entry_id)
            existing = client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
            )
            old_payload = existing[0].payload if existing else {}

            payload = {
                **old_payload,
                "document": lesson,
                "timestamp": timestamp,
            }

            vector = get_embedding(lesson)

            client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as e:
            print(f"[WARN] Failed to update vector store for id {entry_id}: {e}")
            return False

    def delete_lesson(self, entry_id: str) -> bool:
        """Remove entry_id from both the JSON store and Qdrant."""
        data = self._load()
        new_data = [e for e in data if e["id"] != str(entry_id)]

        if len(new_data) == len(data):
            print(f"[WARN] delete_lesson: no entry with id {entry_id}")
            return False

        self._save(new_data)

        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[int(entry_id)]),
            )
            return True
        except Exception as e:
            print(f"[WARN] Failed to delete vector for id {entry_id}: {e}")
            return False