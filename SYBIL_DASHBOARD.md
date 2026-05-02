# 🛡️ SYBIL Mission Control (Ground Truth Dashboard)

Use this dashboard to populate **SYBIL** with instant facts. 

## ⚡ Super-Nitro Injector
Run the script below to insert "Absolute Truths" into MongoDB. These facts **skip LLM reasoning** and play corrections in < 50ms.

### 1. Edit your facts here:
`agent-py/scripts/inject_truths.py`

### 2. Inject them:
Run this command in your terminal to sync the truths to MongoDB Atlas:

```bash
uv run scripts/inject_truths.py
```

---

## 🛠️ Automated Injection Script
Located at `agent-py/scripts/inject_truths.py`.

## 📊 Database Health
- **Knowledge Collection:** `knowledge` (Search Index: `knowledge_embedding_index`)
- **Claims Cache:** `claims` (Search Index: `claims_embedding_index`)
- **Source Trust:** `source_scores` (Weighted ranking for Tavily hits)
