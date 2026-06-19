# NCT to Drug Revenue Linker

This tool matches clinical trials to related FDA-approved prescription drugs and estimates the target indication's average annual revenue.

## Approach

1. **Vector Database Generation**: 
   We fetch unique prescription drug labels from the openFDA API and extract their indications and usage text. We embed these descriptions using Gemini's embedding API (gemini-embedding-001) and store them in a local vector database file (vectordb.pkl).

2. **Trial Indication Retrieval**: 
   Given a trial NCT ID, we query the ClinicalTrials.gov API v2 to retrieve its target conditions.

3. **Semantic Vector Search**: 
   We generate a vector embedding of the trial's target conditions and calculate the cosine similarity against all drug vectors in the local database to retrieve the top 5 most similar FDA-approved drugs.

4. **Revenue Estimation and Averaging**: 
   We query the Gemini API to retrieve or estimate the annual gross revenues for these top 5 matched drugs and calculate the average revenue for the indication.

## Setup

Set your Gemini API key in a .env file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## How to Run

1. **Initialize the Vector Database**:
   ```bash
   python3 populate_vectordb.py
   ```
   This downloads and embeds 500 unique drug labels from openFDA and saves them to vectordb.pkl.

2. **Link a Clinical Trial and Estimate Revenue**:
   ```bash
   python3 nct_revenue_linker.py NCT02993146
   ```

## Sample Output

```text
Fetching trial details for NCT02993146 from ClinicalTrials.gov...

Trial: Ropidoxuridine and Whole Brain Radiation Therapy in Treating Patients With Brain Metastases
Sponsor: National Cancer Institute (NCI)
Conditions: Hematopoietic and Lymphoid Cell Neoplasm, Malignant Solid Neoplasm, Metastatic Malignant Neoplasm in the Brain
Phase: PHASE1

Embedding trial indications and searching vector database...

Top 5 matched FDA approved drugs based on indications:
  1. Temozolomide (temozolomide) - Manufacturer: Bryant Ranch Prepack (Similarity: 0.6018)
  2. Temodar (temozolomide) - Manufacturer: Merck Sharp & Dohme Llc (Similarity: 0.5956)
  3. Cyclophosphamide (cyclophosphamide) - Manufacturer: Xgen Pharmaceuticals Djb, Inc. (Similarity: 0.5935)
  4. Doxorubicin Hydrochloride (doxorubicin hydrochloride) - Manufacturer: Pfizer Laboratories Div Pfizer Inc (Similarity: 0.5677)
  5. Levoleucovorin (levoleucovorin calcium) - Manufacturer: Meitheal Pharmaceuticals Inc. (Similarity: 0.5592)

Querying Gemini for drug revenues...

Estimated Drug Revenues:
  * Temozolomide: $7,500,000.00
  * Temodar: $1,050,000,000.00
  * Cyclophosphamide: $25,000,000.00
  * Doxorubicin Hydrochloride: $75,000,000.00
  * Levoleucovorin: $25,000,000.00

Average Estimated Yearly Revenue for Indication: $236,500,000.00
```
