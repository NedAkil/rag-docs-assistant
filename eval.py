"""Eval: retrieval hit rate + LLM-judged answer correctness over 8 Q&A pairs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import rag  # noqa: E402

# (question, expected source file, key fact the answer must contain / convey, expect_refusal)
CASES = [
    ("How many vacation days do employees get per year?", "employee_handbook.md", "20 vacation days", False),
    ("What is the annual learning budget?", "employee_handbook.md", "1,500 CAD", False),
    ("How much does the Team plan cost per user?", "product_faq.md", "49 USD per user per month", False),
    ("Which data sources does Lumen support?", "product_faq.md", "PostgreSQL, MySQL, BigQuery, Snowflake, CSV", False),
    ("Can I get a refund on an annual plan after 20 days?", "refund_policy.pdf", "yes, full refund within 30 days", False),
    ("How long do refunds take to process?", "refund_policy.pdf", "7 business days", False),
    ("What is Enterprise support response time?", "product_faq.md", "1 hour, 24/7", False),
    ("What is the CEO's favorite color?", None, "refusal", True),
]


def judge(question, expected, got):
    v = rag.complete(
        f"Question: {question}\nExpected key fact: {expected}\nAnswer given: {got}\n"
        "Does the answer correctly convey the expected key fact? Reply with only YES or NO.",
        system="You are a strict grader.")
    return v.strip().upper().startswith("YES")


def main():
    idx = rag.Index.build(save=False)
    hits_ok = ans_ok = n_ret = 0
    for q, src, fact, refuse in CASES:
        r = rag.answer(idx, q)
        if refuse:
            ok = rag.NOT_FOUND.lower().rstrip(".") in r["answer"].lower()
            hit = None
        else:
            n_ret += 1
            hit = src in [c.source for c, _ in r["hits"]]
            hits_ok += hit
            ok = judge(q, fact, r["answer"])
        ans_ok += ok
        print(f"[{'PASS' if ok else 'FAIL'}] hit={hit} | {q}\n        -> {r['answer'][:140]!r}")
    print(f"\nRetrieval hit rate: {hits_ok}/{n_ret} = {hits_ok/n_ret:.0%}")
    print(f"Answer correctness (incl. 1 refusal case): {ans_ok}/{len(CASES)} = {ans_ok/len(CASES):.0%}")


if __name__ == "__main__":
    main()
