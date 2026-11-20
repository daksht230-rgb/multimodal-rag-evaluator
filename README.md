# Multimodal RAG Evaluator

A high-performance Python framework engineered for assessing retrieval grounding and faithfulness across text, table, and chart modalities in RAG architectures.

## Architecture & Core Features
- **Modality Grounding Validation:** Automated heuristics checking for context relevance and answer faithfulness across text, tables, and visual charts.
- **Batch Evaluation & Metrics:** Computes token counts, overlap ratios, and success rate summaries at scale.
- **RESTful Evaluation API:** Built with FastAPI and Pydantic for high-throughput asynchronous evaluation runs.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python main.py
