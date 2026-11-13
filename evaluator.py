import pandas as pd
from typing import List, Dict, Any

class MultimodalRAGEvaluator:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def evaluate_modality_grounding(self, modality_type: str, retrieved_context: str, model_answer: str) -> Dict[str, Any]:
        """Evaluates grounding and faithfulness across text, table, and chart modalities."""
        issues = []
        
        if not retrieved_context or len(retrieved_context.strip()) < 5:
            issues.append(f"Retrieved {modality_type} context is empty or insufficient.")
        if not model_answer or len(model_answer.strip()) < 5:
            issues.append("Model answer is insufficient.")

        # Simulate multimodal faithfulness and relevance scoring heuristics
        context_words = set(retrieved_context.lower().split())
        answer_words = set(model_answer.lower().split())
        
        overlap_ratio = len(context_words.intersection(answer_words)) / max(len(answer_words), 1)
        faithfulness_score = round(min(max(overlap_ratio * 1.5, 0.4), 1.0), 2)
        
        is_grounded = faithfulness_score >= self.threshold and len(issues) == 0

        return {
            "modality": modality_type.upper(),
            "is_grounded": is_grounded,
            "faithfulness_score": faithfulness_score,
            "validation_issues": issues,
            "metrics": {
                "context_token_count": len(retrieved_context.split()),
                "answer_token_count": len(model_answer.split()),
                "overlap_ratio": round(overlap_ratio, 2)
            }
        }

    def evaluate_batch(self, evaluation_items: List[Dict[str, str]]) -> pd.DataFrame:
        results = []
        for idx, item in enumerate(evaluation_items):
            modality = item.get("modality", "text")
            context = item.get("context", "")
            answer = item.get("answer", "")
            
            eval_res = self.evaluate_modality_grounding(modality, context, answer)
            
            results.append({
                "sample_id": f"MM-RAG-{idx+1:03d}",
                "modality": eval_res["modality"],
                "is_grounded": eval_res["is_grounded"],
                "faithfulness_score": eval_res["faithfulness_score"],
                "error_logs": " | ".join(eval_res["validation_issues"]) if eval_res["validation_issues"] else "NONE"
            })
            
        return pd.DataFrame(results)
