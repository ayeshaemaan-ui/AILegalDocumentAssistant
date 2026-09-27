"use client";

import { Card } from "./ui";

/**
 * The maths the system actually runs, with each formula next to the reason it
 * was chosen over the obvious alternative. Formulas are typeset as monospace
 * rather than pulled in through a LaTeX renderer - they are short enough that
 * a 300KB dependency would not earn its place.
 */

type Algo = {
  name: string;
  where: string;
  formula: string;
  plain: string;
  why: string;
};

const ALGORITHMS: Algo[] = [
  {
    name: "Cosine similarity",
    where: "Dense retrieval · BGE embeddings",
    formula: "sim(a, b) =  (a · b) / (‖a‖ ‖b‖)     →  a · b   when both are L2-normalised",
    plain:
      "Measures the angle between two 768-dimensional vectors, ignoring their length.",
    why:
      "Euclidean distance would punish a one-line clause for being shorter than a three-paragraph one about the same obligation. Only the angle carries the meaning. Vectors are normalised at encode time, so the cosine collapses to a plain dot product — one multiply-add per dimension instead of three passes.",
  },
  {
    name: "BM25 (Okapi)",
    where: "Sparse retrieval · exact terms",
    formula:
      "score(D, Q) =  Σ  IDF(qᵢ) · ( f(qᵢ,D) · (k₁+1) ) / ( f(qᵢ,D) + k₁ · (1 − b + b · |D|/avgdl) )",
    plain:
      "Term-frequency scoring with two corrections: saturation (k₁) so the tenth occurrence of a word adds little, and length normalisation (b) so long chunks do not win by being long.",
    why:
      "Embeddings blur identifiers. Ask for \"Siemens\" and dense retrieval happily returns a distributor clause naming someone else — the vector encodes \"a party to a distribution agreement\", not that specific name. BM25 matches the literal token, which is exactly what a party name, a section number, or $5,000,000 needs.",
  },
  {
    name: "Reciprocal Rank Fusion",
    where: "Combining dense + sparse",
    formula: "RRF(d) =  Σ   wᵢ / ( k + rankᵢ(d) )            k = 60",
    plain:
      "Throws away both scoring scales and fuses on rank position alone.",
    why:
      "A cosine of 0.82 and a BM25 score of 14.3 are not comparable — they are not even the same unit, and BM25's magnitude moves with query length. Rank is the one thing both arms agree on. The k=60 constant flattens the top of the curve so rank 1 does not overwhelm rank 3, which lets agreement across arms win: a chunk placed #1 and #30 beats one placed #15 twice.",
  },
  {
    name: "Cross-encoder re-ranking",
    where: "Precision pass over ~20 candidates",
    formula: "score = σ( BERT([CLS] query [SEP] clause [SEP]) )      σ(x) = 1 / (1 + e⁻ˣ)",
    plain:
      "Reads the question and the clause together in one forward pass, with full attention between them.",
    why:
      "The embedding model is a bi-encoder: it encodes query and clause separately, which is what makes searching 40,000 chunks feasible, but it never sees the two texts side by side. So it cannot notice that a clause says \"Supplier may terminate\" when the question asked what Customer may do. The cross-encoder can — at a cost far too high for the whole corpus, so it runs only over what hybrid search already shortlisted. Retrieve wide, re-rank narrow. The sigmoid is display-only: raw logits run about −11..+11, and showing a user \"−3.8\" for a good match is meaningless.",
  },
  {
    name: "Maximal Marginal Relevance",
    where: "Diversity (measured harmful here — off by default)",
    formula: "MMR = argmax [ λ · sim(d, q)  −  (1 − λ) · max sim(d, dⱼ) ]      dⱼ ∈ already selected",
    plain:
      "Each pick trades relevance to the query against redundancy with what is already selected.",
    why:
      "Standard advice, and it measurably hurt on this corpus: −0.014 to −0.024 nDCG in every pairing. The reason is legible — retrieval here is usually scoped to one contract, where near-duplicate clauses are typically all relevant, so buying diversity costs relevance it cannot repay. Kept configurable because the trade-off inverts for unscoped search across all 510 contracts, where a single document can flood the candidate pool.",
  },
  {
    name: "nDCG",
    where: "Evaluation · headline metric",
    formula: "DCG@k = Σ  relᵢ / log₂(i + 1)          nDCG@k = DCG@k / IDCG@k",
    plain:
      "Sums relevance with a logarithmic discount by position, then divides by the best possible ordering of the same results.",
    why:
      "hit_rate says whether the right clause was found; nDCG says whether it was found near the top. That distinction is the whole case for re-ranking, and it is why both are reported. The logarithmic discount matches how attention actually decays — rank 1 to 2 matters far more than rank 9 to 10. Normalising by the ideal ordering keeps it in [0,1] and comparable across questions with different numbers of relevant clauses.",
  },
  {
    name: "Mean Reciprocal Rank",
    where: "Evaluation · position of the first hit",
    formula: "MRR = (1/|Q|) · Σ  1 / rank_of_first_relevant",
    plain:
      "How far down the list the first correct clause appeared, averaged over questions.",
    why:
      "The pair to watch is hit_rate against MRR. High hit_rate with low MRR means the retriever is finding the right clause and burying it — a ranking problem, not a retrieval problem. Those have completely different fixes, and one number cannot tell them apart.",
  },
  {
    name: "Token bucket",
    where: "Rate limiting",
    formula: "tokens = min(burst, tokens + elapsed · rate);   allow  ⟺  tokens ≥ 1",
    plain:
      "A per-client bucket that refills continuously and is drained one token per request.",
    why:
      "A fixed window lets a client fire its full quota at 11:59:59 and again at 12:00:00 — a double burst the limit was supposed to prevent. A bucket refills smoothly, so it absorbs a legitimate burst while still bounding the sustained rate.",
  },
];

export function Algorithms() {
  return (
    <div className="flex flex-col gap-3">
      {ALGORITHMS.map((algo) => (
        <Card key={algo.name} className="p-4">
          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <h3 className="text-[14px] font-semibold">{algo.name}</h3>
            <span className="font-mono text-[11px]" style={{ color: "var(--text-muted)" }}>
              {algo.where}
            </span>
          </div>

          <pre
            className="mt-2.5 overflow-x-auto rounded-lg border px-3 py-2.5 font-mono text-[11.5px] leading-relaxed"
            style={{ borderColor: "var(--border)", background: "var(--surface-sunken)" }}
          >
            {algo.formula}
          </pre>

          <p className="mt-2 text-[12.5px] leading-relaxed">{algo.plain}</p>
          <p className="mt-1.5 text-[12.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
            {algo.why}
          </p>
        </Card>
      ))}
    </div>
  );
}
