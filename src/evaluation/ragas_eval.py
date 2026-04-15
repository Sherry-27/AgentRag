from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset
from src.agent.graph import run_agent

# Create 10-15 test cases from YOUR actual documents
# Replace these with questions from your PDFs
TEST_CASES = [
    {
        "question": "What is the main methodology used in the study?",
        "ground_truth": "Replace this with actual answer from your PDF"
    },
    {
        "question": "What were the key findings or results?",
        "ground_truth": "Replace this with actual answer from your PDF"
    },
    # Add more from your actual PDFs — more = better RAGAS score reliability
]


def run_evaluation():
    questions, answers, contexts, ground_truths = [], [], [], []

    print("Running agent on test cases...")
    for case in TEST_CASES:
        result = run_agent(case["question"])
        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append([doc["text"] for doc in result.get("reranked_docs", [])])
        ground_truths.append(case["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("Evaluating with RAGAS...")
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
    )

    print("\n RAGAS Results:")
    print(f"  Context Precision : {results['context_precision']:.2f}")
    print(f"  Context Recall    : {results['context_recall']:.2f}")
    print(f"  Faithfulness      : {results['faithfulness']:.2f}")
    print(f"  Answer Relevancy  : {results['answer_relevancy']:.2f}")

    return results


if __name__ == "__main__":
    run_evaluation()