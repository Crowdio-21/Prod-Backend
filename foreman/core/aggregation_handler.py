"""Aggregation strategies for DNN branch merge and job completion output."""

from collections import Counter
from typing import Any, Dict, List, Optional


class AggregationHandler:
    """Applies aggregation strategies to a set of task outputs."""

    def aggregate(
        self,
        results: List[Any],
        strategy: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        strategy = (strategy or "average").lower()
        strategy_params = strategy_params or {}

        non_null = [r for r in results if r is not None]
        if not non_null:
            return None

        if strategy == "weighted_sum":
            return self._weighted_sum(non_null, strategy_params)
        if strategy == "voting":
            return self._voting(non_null)
        return self._average(non_null)

    def _average(self, values: List[Any]) -> Any:
        if all(isinstance(v, (int, float)) for v in values):
            return sum(values) / len(values)

        # Common DNN output shape: dict with score/prediction.
        scores = [
            v.get("score")
            for v in values
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float))
        ]
        predictions = [
            v.get("prediction")
            for v in values
            if isinstance(v, dict) and isinstance(v.get("prediction"), str)
        ]

        if scores:
            merged: Dict[str, Any] = {"score": sum(scores) / len(scores)}
            if predictions:
                merged["prediction"] = Counter(predictions).most_common(1)[0][0]
            merged["strategy"] = "average"
            merged["count"] = len(values)
            return merged

        return values

    def _weighted_sum(self, values: List[Any], params: Dict[str, Any]) -> Any:
        weights = params.get("weights", [])
        if not weights or len(weights) != len(values):
            # Fallback to simple average if weights are missing/misaligned.
            return self._average(values)

        if all(isinstance(v, (int, float)) for v in values):
            return sum(v * w for v, w in zip(values, weights))

        scores = [
            v.get("score")
            for v in values
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float))
        ]
        if len(scores) == len(weights):
            return {
                "score": sum(s * w for s, w in zip(scores, weights)),
                "strategy": "weighted_sum",
                "count": len(values),
            }

        return self._average(values)

    def _voting(self, values: List[Any]) -> Any:
        labels = []
        for value in values:
            if isinstance(value, str):
                labels.append(value)
            elif isinstance(value, dict) and isinstance(value.get("prediction"), str):
                labels.append(value.get("prediction"))

        if not labels:
            return self._average(values)

        winner, votes = Counter(labels).most_common(1)[0]
        return {
            "prediction": winner,
            "votes": votes,
            "strategy": "voting",
            "count": len(labels),
        }
