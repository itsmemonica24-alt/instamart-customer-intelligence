from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = """
POLICY ASSISTANT

Answer the question using only the supplied policy context.
Do not invent facts or use knowledge outside the context.
If the supplied context does not contain the answer, your answer must begin:
I could not find
Keep the answer concise and grounded in the supplied policy text.
""".strip()


# ------------------------------------------------------------
# TASK 1: CHUNK POLICY DOCUMENTS
# ------------------------------------------------------------

def load_policy_chunks(overlap=10):
    """Split policy documents into 40-word chunks with the requested overlap."""
    if overlap not in (0, 10):
        raise ValueError("overlap must be either 10 or 0")

    chunks = []
    step = 40 - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        document = path.read_text(encoding="utf-8")
        words = document.split()

        if not words:
            continue

        start = 0
        while start < len(words):
            chunk_words = words[start:start + 40]

            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words),
            })

            # Stop after the first chunk that reaches the end.
            if start + 40 >= len(words):
                break

            start += step

    return chunks


# ------------------------------------------------------------
# TASK 2: TF-IDF VECTORISATION AND RETRIEVAL
# ------------------------------------------------------------

def build_retriever(chunks):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return vectorizer, matrix


def retrieve(query, k=3, chunks=None, vectorizer=None, matrix=None):
    """Return the top-k chunks as dictionaries with source, score and text."""
    if chunks is None:
        chunks = CHUNKS
    if vectorizer is None:
        vectorizer = VECTORIZER
    if matrix is None:
        matrix = CHUNK_MATRIX

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    # Stable tie-breaking by original chunk order.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index)
    )[:k]

    return [
        {
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"],
        }
        for index in ranked_indices
    ]


# ------------------------------------------------------------
# TASK 3: RETRIEVAL-AUGMENTED ANSWERING
# ------------------------------------------------------------

def answer(question, chunks=None, vectorizer=None, matrix=None):
    results = retrieve(
        question,
        k=3,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
    )

    # Each chunk has its source label on a separate line.
    context_parts = []
    for item in results:
        context_parts.append(f"[{item['source']}]\n{item['text']}")

    user_message = (
        "CONTEXT:\n"
        + "\n".join(context_parts)
        + "\nQUESTION:\n"
        + str(question)
    )

    response = mock_llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return {
        "answer": response.choices[0].message.content or "",
        "sources": [item["source"] for item in results],
        "retrieved_chunks": results,
        "tokens_used": response.usage.total_tokens,
    }


# ------------------------------------------------------------
# TASK 4: EVALUATE RETRIEVAL, GROUNDING AND FALLBACK
# ------------------------------------------------------------

def same_document(expected_doc, retrieved_source):
    expected = str(expected_doc).strip().lower()
    source = str(retrieved_source).strip().lower()

    if expected == source:
        return True

    # Also support labels without the ".txt" extension.
    return Path(expected).stem == Path(source).stem


def evaluate(overlap, output_filename):
    chunks = load_policy_chunks(overlap=overlap)
    vectorizer, matrix = build_retriever(chunks)

    evaluation = pd.read_csv(EVAL_FILE, keep_default_na=False)

    required_columns = {"question", "expected_doc", "expected_fact"}
    missing = required_columns - set(evaluation.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in policy_eval.csv: {sorted(missing)}"
        )

    results = []

    for _, row in evaluation.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"])

        result = answer(
            question,
            chunks=chunks,
            vectorizer=vectorizer,
            matrix=matrix,
        )

        retrieved_sources = result["sources"]
        answer_text = result["answer"]

        answerable = expected_doc.upper() != "NONE"

        doc_found = (
            any(same_document(expected_doc, source)
                for source in retrieved_sources)
            if answerable else False
        )

        # The assignment requires an exact, case-sensitive substring check.
        fact_found = (
            expected_fact in answer_text
            if answerable and expected_fact else False
        )

        fallback_found = (
            answer_text.startswith("I could not find")
            if not answerable else False
        )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answerable": answerable,
            "retrieved_sources": ";".join(retrieved_sources),
            "answer": answer_text,
            "expected_doc_retrieved": doc_found,
            "expected_fact_found": fact_found,
            "fallback_correct": fallback_found,
            "tokens_used": result["tokens_used"],
        })

    result_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / output_filename
    result_df.to_csv(output_path, index=False)

    answerable_rows = result_df[result_df["answerable"]]
    unanswerable_rows = result_df[~result_df["answerable"]]

    doc_hits = int(answerable_rows["expected_doc_retrieved"].sum())
    fact_hits = int(answerable_rows["expected_fact_found"].sum())
    fallback_hits = int(unanswerable_rows["fallback_correct"].sum())

    print("\n" + "=" * 68)
    print(f"POLICY RAG EVALUATION — OVERLAP {overlap}")
    print("=" * 68)
    print(f"Policy chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(result_df)}")
    print(
        f"Answerable questions with expected document retrieved: "
        f"{doc_hits}/{len(answerable_rows)}"
    )
    print(
        f"Answerable questions containing expected fact: "
        f"{fact_hits}/{len(answerable_rows)}"
    )
    print(
        f"Unanswerable questions with correct fallback: "
        f"{fallback_hits}/{len(unanswerable_rows)}"
    )
    print(f"Saved detailed results: {output_path}")

    failed = answerable_rows[
        (~answerable_rows["expected_doc_retrieved"])
        | (~answerable_rows["expected_fact_found"])
    ]

    if len(failed):
        print("\nAnswerable questions that failed at least one check:")
        for _, row in failed.iterrows():
            print(f"- {row['question']}")
            print(f"  Expected document: {row['expected_doc']}")
            print(f"  Expected fact: {row['expected_fact']}")
            print(f"  Retrieved: {row['retrieved_sources']}")
            print(f"  Answer: {row['answer']}")
    else:
        print("\nNo answerable questions failed either check.")

    return result_df, chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {EVAL_FILE}")

    # Default retriever used by retrieve() and the later refund-agent script.
    global CHUNKS, VECTORIZER, CHUNK_MATRIX

    CHUNKS = load_policy_chunks(overlap=10)
    VECTORIZER, CHUNK_MATRIX = build_retriever(CHUNKS)

    print("=" * 68)
    print("POLICY DOCUMENT CHUNK AUDIT")
    print("=" * 68)
    print(f"Total chunks with overlap 10: {len(CHUNKS)}")

    counts = pd.Series([chunk["source"] for chunk in CHUNKS]).value_counts()
    for source, count in counts.items():
        print(f"{source}: {count}")

    test_results = retrieve("refund policy late delivery", k=3)
    print("\nTop 3 results for: refund policy late delivery")
    for rank, item in enumerate(test_results, start=1):
        print(
            f"{rank}. {item['source']} "
            f"(score={item['score']:.4f})"
        )

    # Explain TF-IDF as part of the assignment output.
    print("\nTF-IDF explanation:")
    print(
        "Term Frequency (TF) measures how often a term appears in a document "
        "or chunk. Inverse Document Frequency (IDF) gives less weight to "
        "terms common across documents and more weight to terms that are "
        "distinctive."
    )
    print(
        "Limitation: TF-IDF relies on word overlap, so a question and a policy "
        "passage can have the same meaning but rank poorly when they use "
        "different wording."
    )

    # Run the evaluation twice, as required.
    evaluate(overlap=10, output_filename="policy_rag_eval.csv")
    evaluate(overlap=0, output_filename="policy_rag_eval_zero_overlap.csv")


if __name__ == "__main__":
    main()