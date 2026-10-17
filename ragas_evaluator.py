"""
RAGAS evaluation for the NASA mission RAG system.

Metrics
-------
Always computed (no reference answer needed):
- response_relevancy: does the answer address the question? RAGAS asks an LLM to write
  questions that the answer would fit, then compares them with the real question
  using embeddings. 1.0 = fully on topic.
- faithfulness: share of the answer's claims that the retrieved context supports.
  1.0 = every claim is grounded, lower values point to hallucination.

Computed only when a reference (expected) answer is given, as in batch evaluation:
- context_precision: LLM-judged share of retrieved chunks that are useful for reaching
  the reference answer, weighted so that useful chunks ranked higher count more.
- bleu: n-gram overlap between the answer and the reference (0-1, strict word matching).
- rouge_l: longest common word sequence between answer and reference (F-measure, 0-1).

All scores range from 0 to 1. Higher is better.
"""

import argparse
import asyncio
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# RAGAS 0.4 marks the classic metric API used by the starter as deprecated. It still
# works, so these notices are hidden to keep the output readable.
warnings.filterwarnings("ignore", message=r".*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*non-supported Python version.*", category=FutureWarning)

from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.async_utils import run as run_async
    from ragas.metrics import (BleuScore, Faithfulness, LLMContextPrecisionWithReference,
                               ResponseRelevancy, RougeScore)
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "evaluation_dataset.txt"
DEFAULT_RESULTS = PROJECT_DIR / "evaluation_results.csv"

EVALUATOR_MODEL = "gpt-3.5-turbo"
EVALUATOR_EMBEDDING_MODEL = "text-embedding-3-small"

# RAGAS metric names -> names shown in the app and reports.
METRIC_DISPLAY_NAMES = {
    "answer_relevancy": "response_relevancy",
    "faithfulness": "faithfulness",
    "llm_context_precision_with_reference": "context_precision",
    "bleu_score": "bleu",
    "rouge_score": "rouge_l",
}
METRIC_ORDER = list(METRIC_DISPLAY_NAMES.values())

REQUIRED_DATASET_FIELDS = ("id", "category", "question", "expected_answer")


def get_openai_settings() -> Dict[str, Optional[str]]:
    """Read the API key and optional Vocareum base URL from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY in your .env file.")
    return {"api_key": api_key, "base_url": os.getenv("OPENAI_BASE_URL") or None}


def create_evaluator_llm() -> "LangchainLLMWrapper":
    """Create a new evaluator LLM.

    Every metric gets its own instance: RAGAS changes `n` and `temperature` on the LLM
    during a call, so a shared instance gives wrong results when metrics run in parallel.
    """
    return LangchainLLMWrapper(ChatOpenAI(model=EVALUATOR_MODEL, temperature=0, **get_openai_settings()))


@lru_cache(maxsize=1)
def get_evaluator_embeddings() -> "LangchainEmbeddingsWrapper":
    """Create the evaluator embeddings once and reuse them (they hold no per-call state)."""
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EVALUATOR_EMBEDDING_MODEL, **get_openai_settings()))


def build_metrics(with_reference: bool) -> list:
    """Return the metric instances to run."""
    metrics = [
        ResponseRelevancy(llm=create_evaluator_llm(), embeddings=get_evaluator_embeddings()),
        Faithfulness(llm=create_evaluator_llm()),
    ]
    if with_reference:
        metrics += [
            LLMContextPrecisionWithReference(llm=create_evaluator_llm()),
            BleuScore(),
            RougeScore(rouge_type="rougeL"),
        ]
    return metrics


def validate_inputs(question: str, answer: str, contexts: List[str]) -> Optional[str]:
    """Return an error message for bad input, or None when the input is valid."""
    if not isinstance(question, str) or not question.strip():
        return "Question must be a non-empty string."
    if not isinstance(answer, str) or not answer.strip():
        return "Answer must be a non-empty string."
    if answer.startswith("Error generating response"):
        return "Answer is an error message, so it cannot be evaluated."
    if not isinstance(contexts, (list, tuple)):
        return "Contexts must be a list of strings."
    if not [c for c in contexts if isinstance(c, str) and c.strip()]:
        return "No retrieved context to evaluate against."
    return None


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              reference: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics

    Args:
        question: The user's question.
        answer: The generated answer.
        contexts: The retrieved context chunks the answer was based on (a list, or one string).
        reference: Optional expected answer. Enables context_precision, bleu and rouge_l.

    Returns:
        {metric_name: score} with scores from 0 to 1, for example
        {"response_relevancy": 0.91, "faithfulness": 0.85}. On bad input or a failure
        it returns {"error": "<message>"}. Metrics that fail on their own are listed
        under "failed_metrics" while the other scores are still returned.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Accept a single context string as well as a list of chunks.
    if isinstance(contexts, str):
        contexts = [contexts]

    error = validate_inputs(question, answer, contexts)
    if error:
        return {"error": error}

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=[c for c in contexts if isinstance(c, str) and c.strip()],
        reference=reference.strip() if reference and reference.strip() else None,
    )

    try:
        metrics = build_metrics(with_reference=sample.reference is not None)
    except Exception as exc:
        return {"error": f"Could not create evaluator: {exc}"}

    async def score_metric(metric):
        try:
            return metric.name, await metric.single_turn_ascore(sample), None
        except Exception as exc:
            return metric.name, None, str(exc)

    async def score_all():
        return await asyncio.gather(*(score_metric(metric) for metric in metrics))

    # All metrics run at the same time in one event loop. They share one OpenAI
    # client, which must not be used from several event loops or threads at once.
    outcomes = run_async(score_all)

    scores: Dict[str, float] = {}
    failures = []
    for name, value, failure in outcomes:
        display_name = METRIC_DISPLAY_NAMES.get(name, name)
        if failure is None and value is not None and not math.isnan(value):
            scores[display_name] = round(float(value), 4)
        else:
            failures.append(f"{display_name}: {failure or 'no score returned'}")

    if not scores:
        return {"error": "All metrics failed. " + "; ".join(failures)}
    if failures:
        scores["failed_metrics"] = "; ".join(failures)
    return scores


def load_evaluation_dataset(path: Path = DEFAULT_DATASET) -> List[Dict[str, str]]:
    """Read the '---'-separated 'key: value' blocks of the evaluation dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    items = []
    blocks = path.read_text(encoding="utf-8").split("\n---")
    for block_number, block in enumerate(blocks, start=1):
        item = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            item[key.strip().lower()] = value.strip()
        if not item:
            continue
        missing = [field for field in REQUIRED_DATASET_FIELDS if not item.get(field)]
        if missing:
            raise ValueError(f"Block {block_number} in {path.name} is missing: {', '.join(missing)}")
        item.setdefault("mission", "all")
        items.append(item)

    if not items:
        raise ValueError(f"No questions found in {path}")
    return items


def run_batch_evaluation(dataset_path: Path = DEFAULT_DATASET,
                         chroma_dir: str = "./chroma_db_openai",
                         collection_name: str = "nasa_space_missions_text",
                         top_k: int = 5,
                         model: str = "gpt-3.5-turbo",
                         use_mission_filter: bool = True,
                         results_path: Optional[Path] = DEFAULT_RESULTS):
    """Run every test question through retrieval, generation and RAGAS scoring.

    Prints the scores per question and the mean of each metric, and returns a DataFrame.
    """
    import pandas as pd

    import llm_client
    import rag_client

    items = load_evaluation_dataset(dataset_path)
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        raise RuntimeError(f"Could not open ChromaDB collection '{collection_name}': {error}")

    rows = []
    for number, item in enumerate(items, start=1):
        print(f"[{number}/{len(items)}] {item['id']}: {item['question']}")
        mission_filter = item["mission"] if use_mission_filter else None

        row = {"id": item["id"], "category": item["category"], "mission": item["mission"],
               "question": item["question"]}
        try:
            results = rag_client.retrieve_documents(collection, item["question"], top_k, mission_filter)
            documents = results["documents"][0]
            context = rag_client.format_context(documents, results["metadatas"][0],
                                                results["distances"][0])
            answer = llm_client.generate_response(None, item["question"], context, [], model)
            scores = evaluate_response_quality(item["question"], answer, documents,
                                               reference=item["expected_answer"])
            row["answer"] = answer
        except Exception as exc:
            scores = {"error": str(exc)}
            row["answer"] = ""

        if "error" in scores:
            print(f"    evaluation error: {scores['error']}")
            row["error"] = scores["error"]
        else:
            print("    " + "  ".join(f"{name}={scores[name]:.3f}" for name in METRIC_ORDER if name in scores))
            if "failed_metrics" in scores:
                print(f"    failed metrics: {scores['failed_metrics']}")
        row.update({name: scores.get(name) for name in METRIC_ORDER})
        rows.append(row)

    results_df = pd.DataFrame(rows)
    print_batch_summary(results_df)

    if results_path:
        results_df.to_csv(results_path, index=False)
        print(f"\nFull results (with answers) saved to {results_path}")
    return results_df


def print_batch_summary(results_df) -> None:
    """Print a per-question score table and the mean of each metric."""
    metric_columns = [name for name in METRIC_ORDER if name in results_df and results_df[name].notna().any()]
    table = results_df[["id", "category", *metric_columns]].copy()
    means = results_df[metric_columns].astype(float).mean()

    print("\n" + "=" * 100)
    print("BATCH EVALUATION RESULTS")
    print("=" * 100)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}", na_rep="-"))
    print("-" * 100)
    print("Mean score per metric:")
    for name, value in means.items():
        print(f"  {name:<20} {value:.3f}")
    print(f"Questions evaluated: {int(results_df[metric_columns[0]].notna().sum()) if metric_columns else 0}"
          f" / {len(results_df)}")


def main():
    """Command-line entry point for batch evaluation."""
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation of the NASA RAG system")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Evaluation dataset file")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per question")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model that writes the answers")
    parser.add_argument("--no-mission-filter", action="store_true",
                        help="Search all missions instead of each question's mission")
    parser.add_argument("--output", default=str(DEFAULT_RESULTS), help="CSV file for full results")
    args = parser.parse_args()

    if not RAGAS_AVAILABLE:
        raise SystemExit("RAGAS is not installed. Run: pip install -r requirements.txt")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        run_batch_evaluation(
            dataset_path=Path(args.dataset),
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name,
            top_k=args.top_k,
            model=args.model,
            use_mission_filter=not args.no_mission_filter,
            results_path=Path(args.output),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Batch evaluation stopped: {exc}")


if __name__ == "__main__":
    main()
