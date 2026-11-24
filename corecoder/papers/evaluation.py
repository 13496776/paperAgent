"""Metrics for labeled intent, rewrite and retrieval datasets."""


def evaluate(rows):
    if not rows:
        raise ValueError("evaluation dataset is empty")
    fields = ("intent", "expected_intent", "query", "expected_query", "relevant_ids", "retrieved_ids")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"evaluation row {index} must be an object")
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"evaluation row {index} missing fields: {', '.join(missing)}")
    intent = sum(row["intent"] == row["expected_intent"] for row in rows) / len(rows)
    rewrite = sum(row["query"].strip().casefold() == row["expected_query"].strip().casefold() for row in rows) / len(rows)
    recalls = []
    for row in rows:
        relevant = set(row["relevant_ids"])
        if relevant:
            recalls.append(len(relevant.intersection(row["retrieved_ids"])) / len(relevant))
    return {"intent_accuracy": intent, "rewrite_exact_match": rewrite,
            "retrieval_recall": sum(recalls) / len(recalls) if recalls else None,
            "examples": len(rows), "recall_examples": len(recalls)}
