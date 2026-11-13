from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from evaluator import MultimodalRAGEvaluator

app = FastAPI(
    title="Multimodal RAG Evaluator",
    description="Production-grade evaluation framework for assessing text, table, and chart grounding accuracy in RAG systems.",
    version="1.0.0"
)

evaluator = MultimodalRAGEvaluator()

class EvaluationItem(BaseModel):
    modality: str = Field(..., example="chart")
    context: str = Field(..., example="Chart shows Q3 revenue increased by 25% reaching $1.5M.")
    answer: str = Field(..., example="Based on the chart, Q3 revenue grew by 25% to $1.5M.")

class BatchEvaluationRequest(BaseModel):
    items: List[EvaluationItem]

@app.post("/v1/eval/multimodal", summary="Evaluate Multimodal RAG Grounding")
def evaluate_multimodal_rag(payload: BatchEvaluationRequest):
    try:
        raw_items = [
            {"modality": item.modality, "context": item.context, "answer": item.answer}
            for item in payload.items
        ]
        
        df_results = evaluator.evaluate_batch(raw_items)
        
        grounded_count = int((df_results["is_grounded"] == True).sum())
        total_count = len(raw_items)
        
        return {
            "status": "success",
            "summary": {
                "total_evaluated": total_count,
                "grounded_samples": grounded_count,
                "grounding_success_rate_pct": round((grounded_count / total_count) * 100, 2) if total_count > 0 else 0.0
            },
            "evaluation_records": df_results.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
