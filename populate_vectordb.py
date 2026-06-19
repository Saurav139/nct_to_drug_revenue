#!/usr/bin/env python3
import os
import sys
import json
import pickle
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Any

# Load .env file manually if exists
def load_env():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_env()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def make_http_request(url: str, method: str = "GET", data: bytes = None, headers: Dict[str, str] = None) -> str:
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")

def make_http_request_with_retry(url: str, method: str = "GET", data: bytes = None, headers: Dict[str, str] = None, max_retries: int = 7) -> str:
    delay = 6
    for attempt in range(max_retries):
        try:
            return make_http_request(url, method, data, headers)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                print(f"Rate limited (429). Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def fetch_openfda_prescription_drugs(limit: int = 500) -> List[Dict[str, Any]]:
    print(f"Fetching up to {limit} prescription drug labels from openFDA...")
    drugs = []
    seen = set()
    skip = 0
    page_size = 100
    
    while len(drugs) < limit:
        current_limit = min(page_size, limit - len(drugs))
        query = 'openfda.product_type.exact:"HUMAN PRESCRIPTION DRUG"'
        encoded_query = urllib.parse.quote(query)
        url = f'https://api.fda.gov/drug/label.json?search={encoded_query}&limit={current_limit}&skip={skip}'
        try:
            raw_json = make_http_request_with_retry(url)
            data = json.loads(raw_json)
        except Exception as e:
            print(f"Error calling openFDA at skip={skip}: {e}")
            break
            
        results = data.get("results", [])
        if not results:
            break
            
        for r in results:
            openfda = r.get("openfda", {})
            brand_names = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            
            if not brand_names or not generic_names:
                continue
                
            brand = brand_names[0].strip().title()
            generic = generic_names[0].strip().lower()
            manufacturer = openfda.get("manufacturer_name", ["Unknown"])[0].strip().title()
            
            # Use combination of brand and generic as unique key
            key = f"{brand}:{generic}"
            if key in seen:
                continue
                
            indications_list = r.get("indications_and_usage", [])
            if not indications_list:
                continue
                
            indications_text = " ".join(indications_list) if isinstance(indications_list, list) else str(indications_list)
            # Remove redundant formatting or truncate if extremely long
            indications_text = indications_text[:1200]
            
            seen.add(key)
            drugs.append({
                "brand_name": brand,
                "generic_name": generic,
                "manufacturer": manufacturer,
                "indications_and_usage": indications_text
            })
            
        skip += len(results)
        print(f"Collected {len(drugs)} unique prescription drugs...")
        
    return drugs[:limit]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch calls Gemini API to embed texts."""
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set the key in your environment or a .env file.", file=sys.stderr)
        sys.exit(1)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"
    
    # We batch requests in chunks of 100 to stay within payload limits
    embeddings = []
    chunk_size = 100
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        requests_payload = []
        for text in chunk:
            requests_payload.append({
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [{"text": text}]
                }
            })
        
        payload = {"requests": requests_payload}
        data_bytes = json.dumps(payload).encode("utf-8")
        
        try:
            raw_response = make_http_request_with_retry(
                url, 
                method="POST", 
                data=data_bytes, 
                headers={"Content-Type": "application/json"}
            )
            res_data = json.loads(raw_response)
            chunk_embeddings = [emb["values"] for emb in res_data.get("embeddings", [])]
            embeddings.extend(chunk_embeddings)
            print(f"Embedded {len(embeddings)} of {len(texts)} indications...")
            # Small delay to avoid API rate limits
            time.sleep(6)
        except Exception as e:
            print(f"Error generating embeddings for chunk starting at {i}: {e}", file=sys.stderr)
            # Add dummy vectors on failure to preserve indexes or abort
            sys.exit(1)
            
    return embeddings

def main():
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment or .env file.")
        print("Please export GEMINI_API_KEY or add it to a .env file in this directory.")
        sys.exit(1)
        
    # 1. Fetch prescription drugs
    drugs = fetch_openfda_prescription_drugs(limit=500)
    if not drugs:
        print("No drugs fetched. Exiting.")
        sys.exit(1)
        
    # 2. Embed indications
    print("Generating vector embeddings using Gemini API...")
    texts_to_embed = [d["indications_and_usage"] for d in drugs]
    embeddings = embed_texts(texts_to_embed)
    
    # 3. Store in database
    vectordb = []
    for drug, emb in zip(drugs, embeddings):
        vectordb.append({
            "brand_name": drug["brand_name"],
            "generic_name": drug["generic_name"],
            "manufacturer": drug["manufacturer"],
            "indications_and_usage": drug["indications_and_usage"],
            "embedding": emb
        })
        
    db_path = "vectordb.pkl"
    with open(db_path, "wb") as f:
        pickle.dump(vectordb, f)
        
    print(f"\nVector database successfully populated with {len(vectordb)} drugs and saved to {db_path}!")

if __name__ == "__main__":
    main()
