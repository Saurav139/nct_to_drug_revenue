#!/usr/bin/env python3
import os
import sys
import json
import pickle
import re
import urllib.request
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

def fetch_trial_data(nct_id: str) -> Dict[str, Any]:
    """Retrieves basic trial details from ClinicalTrials.gov."""
    nct_id = nct_id.strip().upper()
    print(f"Fetching trial details for {nct_id} from ClinicalTrials.gov...")
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        raw_json = make_http_request(url, headers={"Accept": "application/json"})
        data = json.loads(raw_json)
    except Exception as e:
        raise ValueError(f"Failed to fetch trial details: {e}")
        
    protocol = data.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    design_module = protocol.get("designModule", {})
    
    return {
        "nct_id": nct_id,
        "brief_title": id_module.get("briefTitle", "Unknown Title"),
        "lead_sponsor": protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "Unknown Sponsor"),
        "conditions": conditions_module.get("conditions", []),
        "phases": design_module.get("phases", []),
        "overall_status": status_module.get("overallStatus", "UNKNOWN")
    }

def embed_text(text: str) -> List[float]:
    """Generates embedding vector for a given text using Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {
            "parts": [{"text": text}]
        }
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    raw_response = make_http_request(url, method="POST", data=data_bytes, headers={"Content-Type": "application/json"})
    res_data = json.loads(raw_response)
    return res_data["embedding"]["values"]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a * a for a in v1) ** 0.5
    mag2 = sum(a * a for a in v2) ** 0.5
    if mag1 * mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def find_top_matches(trial_embedding: List[float], db_path: str = "vectordb.pkl", top_n: int = 5) -> List[Dict[str, Any]]:
    """Loads the vector database and matches the trial embedding against drug embeddings."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector database not found at {db_path}. Please run populate_vectordb.py first.")
        
    with open(db_path, "rb") as f:
        vectordb = pickle.load(f)
        
    scored_drugs = []
    for drug in vectordb:
        score = cosine_similarity(trial_embedding, drug["embedding"])
        scored_drugs.append((score, drug))
        
    # Sort by score descending
    scored_drugs.sort(key=lambda x: x[0], reverse=True)
    
    top_matches = []
    for score, drug in scored_drugs[:top_n]:
        matched_drug = drug.copy()
        matched_drug["similarity_score"] = score
        # Remove raw embedding vector from print dictionary
        matched_drug.pop("embedding", None)
        top_matches.append(matched_drug)
        
    return top_matches

def estimate_revenue_llm(drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Queries Gemini to estimate the revenue for each of the top matched drugs."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    drugs_info = "\n".join([
        f"- Brand: {d['brand_name']}, Generic: {d['generic_name']}, Manufacturer: {d['manufacturer']}"
        for d in drugs
    ])
    
    prompt = f"""You are a pharmaceutical commercialization analyst. 
For the following 5 FDA-approved prescription drugs, estimate or retrieve their annual gross revenue (or peak annual revenue) in USD.
Use your pre-trained knowledge of drug markets and financial history to provide the best estimate possible.

DRUGS:
{drugs_info}

Output your response as a valid JSON array of objects, containing no other text or markdown fences. Each object must match this schema:
{{
  "brand_name": "string",
  "estimated_revenue_usd": float
}}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    raw_response = make_http_request(url, method="POST", data=data_bytes, headers={"Content-Type": "application/json"})
    res_data = json.loads(raw_response)
    raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    # Clean JSON fences if the model outputted them
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    clean_json = match.group(1).strip() if match else raw_text
    
    return json.loads(clean_json)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 nct_revenue_linker.py <NCT_ID>")
        sys.exit(1)
        
    nct_id = sys.argv[1]
    
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY environment variable is not set. Please export it or add it to a .env file.")
        sys.exit(1)
        
    # 1. Fetch Trial Details
    try:
        trial = fetch_trial_data(nct_id)
    except Exception as e:
        print(f"Error fetching trial: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nTrial: {trial['brief_title']}")
    print(f"Sponsor: {trial['lead_sponsor']}")
    print(f"Conditions: {', '.join(trial['conditions'])}")
    print(f"Phase: {', '.join(trial['phases']) or 'N/A'}")
    
    if not trial["conditions"]:
        print("No trial conditions/indications found to embed. Exiting.")
        sys.exit(1)
        
    # 2. Embed conditions and match
    print("\nEmbedding trial indications and searching vector database...")
    trial_text = " ".join(trial["conditions"])
    trial_emb = embed_text(trial_text)
    
    try:
        matches = find_top_matches(trial_emb, db_path="vectordb.pkl", top_n=5)
    except Exception as e:
        print(f"Vector search failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\nTop 5 matched FDA approved drugs based on indications:")
    for idx, d in enumerate(matches, 1):
        print(f"  {idx}. {d['brand_name']} ({d['generic_name']}) - Manufacturer: {d['manufacturer']} (Similarity: {d['similarity_score']:.4f})")
        
    # 3. Estimate Revenue via LLM
    print("\nQuerying Gemini for drug revenues...")
    try:
        revenues = estimate_revenue_llm(matches)
        
        # Build map for display
        rev_map = {r["brand_name"].lower(): r["estimated_revenue_usd"] for r in revenues}
        
        print("\nEstimated Drug Revenues:")
        total_revenue = 0.0
        count = 0
        
        for d in matches:
            brand_lower = d["brand_name"].lower()
            rev = rev_map.get(brand_lower, 0.0)
            
            # Try to match substring if exact brand name key matches fails
            if rev == 0.0:
                for k, v in rev_map.items():
                    if k in brand_lower or brand_lower in k:
                        rev = v
                        break
                        
            print(f"  * {d['brand_name']}: ${rev:,.2f}")
            total_revenue += rev
            count += 1
            
        avg_revenue = total_revenue / count if count > 0 else 0.0
        print(f"\nAverage Estimated Yearly Revenue for Indication: ${avg_revenue:,.2f}")
        
    except Exception as e:
        print(f"Revenue estimation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
