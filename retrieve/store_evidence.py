import os
import json
from qdrant_client.http import models
from models import embed_model, client
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE_PATH = BASE_DIR.parent / "context" / "evidence.json"

if not os.path.exists(JSON_FILE_PATH):
    print(f"❌ Error: {JSON_FILE_PATH} not found.")
    exit()

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    print("❌ JSON file must contain a list of strings.")
    exit()

domain_knowledge = [str(item).strip() for item in data if str(item).strip()]

print(f"📄 Loaded {len(domain_knowledge)} evidence entries.")


COLLECTION_NAME = "telco_domain_evidence"

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

points = []

for idx, fact in enumerate(domain_knowledge):
    vector = embed_model.embed_query(fact)
    
    points.append(
        models.PointStruct(
            id=idx,
            vector=vector,
            payload={"document": fact}
        )
    )

if points:
    BATCH_SIZE = 5

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        print(f"⏳ Inserting batch {i//BATCH_SIZE + 1}")

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
    print("✅ All batches inserted successfully.")
else:
    print("⚠️ No valid data found in JSON file.")


# TEST RETRIEVAL

test_question = "Churn"
print(f"\n🔍 Testing Retrieval for: '{test_question}'")

query_vector = embed_model.embed_query(test_question)

response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=2
)

results = response.points

if results:
    for i, match in enumerate(results):
        matched_evidence = match.payload["document"]
        print(f"RETRIEVED FACT {i+1}: {matched_evidence} (Score: {match.score:.4f})")
else:
    print("No match found.")

client.close()