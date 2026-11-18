# Phase 6 Evaluation Report

Generated: 2026-11-18T22:11:39+00:00

## 1. Systems Compared

- **System C**: Full system (orchestrator + sub-agents + critic)

## 2. Task Success by System and Category

| System | Overall (Strict) | Overall (Excl. Infra) | Infra Failures | Documents | Database | Combined | Refusal / Safety |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System C** | 100.0% (8/8) | 100.0% | 0 | 100.0% (2/2) | 100.0% (2/2) | 100.0% (2/2) | 100.0% (2/2) |

## 3. SQL Execution Accuracy, Combined Precision/Recall, and Refusal Accuracy

| System | SQL Execution Accuracy | Combined Set Precision | Combined Set Recall | Combined Set F1 | Refusal Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System C** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## 4. Latency, Cost, LLM Calls, and Tokens per Request

| System | Latency p50 (s) | Latency p95 (s) | Avg Cost / Req (USD) | Avg LLM Calls | Avg Input Tokens | Avg Output Tokens | Avg Thinking Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **System C** | 50.8s | 144.0s | $0.00952 | 6.1 | 6290 | 542 | 740 |

## 5. Retrieval Ablation (Document Questions, No LLM)

| Retriever | Questions | Hit@3 | Hit@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Vector only (ChromaDB) | 12 | 91.7% | 91.7% | 0.917 |
| BM25 only (rank_bm25) | 12 | 91.7% | 91.7% | 0.875 |
| Hybrid (RRF) | 12 | 91.7% | 100.0% | 0.896 |

## 6. LLM Judge Verdicts (for Spot-Checking)

| System | ID | Category | Judge Verdict | Judge Reason |
| :--- | :--- | :--- | :---: | :--- |
| C | `doc_01` | documents | PASS | The system answer accurately provides the maximum acceptable return rate of 11.0% for Jeans as specified in the ground truth. |
| C | `doc_02` | documents | PASS | The system answer correctly identifies the 30-day deadline for a monetary refund and notes that returns up to 60 days receive store credit only, matching all required ground truth elements. |

## 7. Failed Questions

All evaluated questions passed.
