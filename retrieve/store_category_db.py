import json
import os
from qdrant_client.http import models
from models import client, get_embedding
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

COLLECTION_NAME = "telco_distinct_values"

BASE_DIR = Path(__file__).resolve().parent
JSON_DATA_PATH = BASE_DIR.parent / "context" / "db_values.json"

# ==========================================
# 2. SETUP QDRANT & COLLECTION
# ==========================================


if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"🗑️ Deleted old collection '{COLLECTION_NAME}'")

print("⏳ Determining vector size from Ollama...")
dummy_vector = get_embedding("test")
vector_size = len(dummy_vector)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
    )
)
print(f"✅ Created collection '{COLLECTION_NAME}' with dimension {vector_size}.")

# ==========================================
# 3. EXTRACTION FROM JSON & INDEXING
# ==========================================

if not os.path.exists(JSON_DATA_PATH):
    print(f"❌ Error: {JSON_DATA_PATH} not found.")
    exit()

with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
    tables_data = json.load(f)

points = []
global_counter = 0

for table in tables_data:
    table_name = table.get("table_name", "").strip()
    columns = table.get("columns", [])

    print(f"\n📊 Processing table: {table_name}")

    for col in columns:
        col_name = col.get("column_name", "").strip()
        values = col.get("values", [])

        if not col_name:
            continue

        print(f"   ➤ Column '{col_name}' → {len(values)} values")

        for val in values:
            if val is None or not str(val).strip():
                continue

            val_str = str(val).strip()

            vector = get_embedding(val_str)

            payload = {
                "document": val_str,
                "table_name": table_name,
                "column_name": col_name,
                "value": val_str
            }

            points.append(
                models.PointStruct(
                    id=global_counter,
                    vector=vector,
                    payload=payload
                )
            )

            global_counter += 1

# ==========================================
# 4. STORE IN VECTOR DB
# ==========================================

if points:
    print(f"\n⏳ Storing {len(points)} values into Qdrant...")
    
    BATCH_SIZE = 50 
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
    print("✅ Value Indexing Complete.")
else:
    print("⚠️ No values found in JSON to index.")

# ==========================================
# 5. TEST RETRIEVAL (Sanity Check)
# ==========================================

test_term = "GAS"
print(f"\n🔍 Testing Retrieval for user term: '{test_term}'")

query_vector = get_embedding(test_term)

response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5
)

results = response.points

if results:
    for i, match in enumerate(results):
        match_payload = match.payload

        print(f"\n--- Match {i+1} (Score: {match.score:.4f}) ---")
        print(f"Document       : {match_payload['document']}")
        print(f"Table          : {match_payload['table_name']}")
        print(f"Column         : {match_payload['column_name']}")
        print(f"Value          : {match_payload['value']}")
else:
    print("No match found.")

client.close()