import json
import os
import requests
import uuid
from models import embed_model, client
from qdrant_client.http import models
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JSON_SCHEMA_PATH = BASE_DIR.parent / "context" / "db_schema.json"

if not os.path.exists(JSON_SCHEMA_PATH):
    print(f"❌ File {JSON_SCHEMA_PATH} not found.")
    exit()

with open(JSON_SCHEMA_PATH, "r", encoding="utf-8") as f:
    tables_json = json.load(f)

if not isinstance(tables_json, list):
    print("❌ JSON must contain a list of tables.")
    exit()

print(f"📄 Loaded {len(tables_json)} tables from JSON.")

COLLECTION_NAME = "telco_db_schema"

if not client.collection_exists(COLLECTION_NAME):
    print("⏳ Determining vector size from Ollama...")
    dummy_vector = embed_model.embed_query("test")
    vector_size = len(dummy_vector)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    print(f"✅ Created collection '{COLLECTION_NAME}' with dimension {vector_size}.")


print("⏳ Generating embeddings and populating Vector DB...")

points = []

for table in tables_json:
    table_name = table.get("table_name", "").strip()
    description = table.get("description", "").strip()
    columns = table.get("columns", [])
    
    if not table_name:
        continue

    col_details_list = []
    for col in columns:
        col_name = col.get("column_name", "").strip()
        col_desc = col.get("description", "").strip()
        col_type = col.get("datatype", "").strip()

        if col_name:
            col_details_list.append(
                f"{col_name} ({col_type}): {col_desc}"
            )

    col_details = ", ".join(col_details_list)

    text_content = (
        f"Table: {table_name}. "
        f"Description: {description}. "
        f"Columns: {col_details}. "
    )
    vector = embed_model.embed_query(text_content)

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, table_name))

    payload = {
        "table_name": table_name,
        "description": description,
        "columns": json.dumps(columns),
        "document": text_content
    }

    points.append(
        models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
    )

BATCH_SIZE = 1

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

print(f"✅ Successfully stored {len(points)} tables in Vector DB.")

# TEST QUERY

query_text = "Customers activation"
print(f"\n🔍 Testing Query: '{query_text}'")

query_vector = embed_model.embed_query(query_text)

response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)

results = response.points

if results:
    top_match = results[0]
    print(f"🏆 Top Match Table: {top_match.payload['table_name']}")
    print(f"📄 Description: {top_match.payload['description']}")
    print(f"📊 Similarity Score: {top_match.score:.4f}")
    client.close()
else:
    print("No match found.")