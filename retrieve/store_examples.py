import json
import os
from qdrant_client.http import models
from models import embed_model, client
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
JSON_FILE_PATH = BASE_DIR.parent / "context" / "question-example.json"

if not os.path.exists(JSON_FILE_PATH):
    print(f"❌ Error: {JSON_FILE_PATH} not found.")
    exit()

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    examples_data = json.load(f)

print(f"📄 Loaded {len(examples_data)} examples from JSON.")

COLLECTION_NAME = "sql_few_shot_examples"

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

# ==========================================
# 3. POPULATE DATABASE
# ==========================================
print("⏳ Generating embeddings and populating Vector DB...")

points = []

for idx, ex in enumerate(examples_data):
    question = ex.get("question", "").strip()
    sql_query = ex.get("sql", "").strip()
    
    if not question:
        continue
        
    vector = embed_model.embed_query(question)

    payload = {
        "document": question,
        "query": sql_query
    }
    
    points.append(
        models.PointStruct(
            id=idx,
            vector=vector,
            payload=payload
        )
    )

if points:
    BATCH_SIZE = 5
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )
    print(f"✅ Successfully stored {len(points)} examples.")
else:
    print("⚠️ No valid data found in JSON file.")

# ==========================================
# 4. TEST RETRIEVAL
# ==========================================
new_user_question = "What is the total number of Revenue for each customer segment?"
print(f"\n🔍 Testing Retrieval for: '{new_user_question}'")

query_vector = embed_model.embed_query(new_user_question)

response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=1
)

results = response.points

if results:
    top_match = results[0]
    print(f"\n🔍 Matched: {top_match.payload['document']}")
    print(f"💻 SQL: {top_match.payload['query']}")
    print(f"📊 Similarity Score: {top_match.score:.4f}")
else:
    print("No match found.")

client.close()