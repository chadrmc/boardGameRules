import anthropic
import json
from models import SearchResult

client = anthropic.Anthropic()

RERANK_PROMPT = """A user searched a board game rulebook for: "{query}"

Here are the candidate elements found:
{candidates}

Determine what the user actually wants, then re-rank from most to least relevant.

Source authority (always apply regardless of intent):
- "expansion" and "core" are equal authority
- "variant" is lowest authority — only rank a variant result highly if the query explicitly mentions a variant (e.g. "amigo variant", "hidden tiger variant", "variant rule for X"). Otherwise push variants to the bottom even if topically relevant.

Intent guide (apply after source authority):
- "example of X" / "show me X" / "what does X look like" → user wants notes, illustrations, or examples that DEMONSTRATE X, not the rule text defining X. Rank supporting elements ABOVE the rule itself.
- "what is the rule for X" / "how does X work" / "explain X" / "describe X" / "how do I X" → user wants the rule element itself first
- "X rule" (bare topic query) → rule element first, supporting material after

Return ONLY a JSON object:
{{
  "ranked_indices": [<0-based indices from most to least relevant>],
  "reasoning": "<one sentence on what the user wants>"
}}"""


def rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    if len(results) <= 1:
        return results

    candidates = "\n".join(
        f"{i}. [{r.element.source_type}:{r.element.type}] {r.element.label}: {r.element.description[:120]}"
        for i, r in enumerate(results)
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": RERANK_PROMPT.format(query=query, candidates=candidates),
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    ranked_indices = data["ranked_indices"]

    # Return results in re-ranked order, ignoring any out-of-range indices
    seen = set()
    reranked = []
    for i in ranked_indices:
        if 0 <= i < len(results) and i not in seen:
            reranked.append(results[i])
            seen.add(i)

    # Append any results Claude didn't mention
    for i, r in enumerate(results):
        if i not in seen:
            reranked.append(r)

    return reranked
