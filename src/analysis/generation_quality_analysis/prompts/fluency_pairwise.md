You are comparing two model responses only on FLUENCY.

Fluency means whether each response is grammatical, coherent, and natural to read. Judge only the form of the writing — how it reads — not its content, topic, accuracy, relevance, genre appropriateness, persuasiveness, creativity, or factuality.

Your task is to decide which response is more fluent.

Rate each response on this 1–5 fluency scale:

1 — Severely non-fluent.
The response is so broken, garbled, or incoherent that it is very hard to read at all.

2 — Mostly non-fluent.
The response has frequent grammar, syntax, wording, or coherence problems that make reading effortful throughout.

3 — Somewhat fluent.
The response is understandable, but noticeable problems make the reader slow down, re-read, or lose the thread in places.

4 — Fluent.
The response may contain minor awkwardness or small errors, but nothing substantially slows reading or impedes comprehension.

5 — Fully fluent.
The response reads naturally from start to finish; grammar, syntax, phrasing, and flow do not make the reader pause.

What should make a response more fluent:
- It is grammatical and easy to read.
- Sentences are well formed.
- The wording is natural rather than awkward.
- The flow between sentences is coherent.
- The reader can follow the text without slowing down or re-reading.
- The text avoids garbled, corrupted, or non-prose stretches.

What should make a response less fluent:
- Grammatical errors, broken syntax, run-on syntax, missing words, or agreement errors.
- Awkward or unnatural phrasing.
- Choppy, disjointed, or hard-to-follow flow.
- Garbled text, corrupted text, stray quotation marks or backticks, visible escape characters such as "\n" or "\"", or dumped prompt/instruction fragments.
- Repetition or looping that disrupts readability.
- Abrupt incoherence that makes the reader lose the thread.

Focus only on fluency. Do NOT judge:
- Prompt relevance.
- Whether the response answers the user's request.
- Factual accuracy.
- Domain correctness.
- Persuasiveness.
- Creativity.
- Helpfulness.
- Informativeness.
- Depth.
- Specificity.
- Verbosity.
- Whether the response is too long or too short.
- Whether the response sounds formal, academic, polished, casual, conversational, or human-written.

Important clarification about relevance:
Do not penalize a response for being off-topic, incomplete, or failing the prompt unless that failure creates a fluency problem in the visible text itself. Relevance is evaluated separately.

Important clarification about factuality:
Do not penalize false, implausible, or unsupported claims when judging fluency. A factually wrong sentence can still be fluent if it reads smoothly.

Important clarification about register and style:
Do not penalize informal, colloquial, plain, simple, or stylistically unusual prose if it is grammatical and easy to read. "Not academic enough," "not polished enough," or "too casual" is not a fluency defect.

Important clarification about technical density:
Dense, jargon-heavy, or specialized prose, such as a scientific abstract, can be fully fluent if it is grammatical and coherent. Do not treat appropriate technical language as non-fluent merely because it is harder to understand.

Important clarification about formatting:
Intentional, well-formed headings, lists, markdown, or paragraph breaks are formatting choices, not fluency defects. However, corrupted markup, stray symbols, or dumped prompt text that disrupts reading should count against fluency.

Important clarification about truncation:
Judge only the visible content. Do not speculate about what either response might have said after truncation. Do not penalize an abrupt ending by itself. However, if the visible content is already hard to read, incoherent, or garbled, assign a lower fluency score accordingly.

Important clarification about minor issues:
A single awkward phrase, or a few scattered minor slips that you can read straight past, should usually receive a 4, not a 3. A score of 3 requires problems that actually make the reader slow down, re-read, or lose the thread.

Important clarification about ties:
Use "tie" when both responses are similarly fluent or similarly non-fluent.
Do not force a winner based on relevance, factuality, style, creativity, polish, length, specificity, or domain quality.
If both responses are fully fluent, choose "tie".
If both responses have comparable fluency problems, choose "tie".

First assess Response A's fluency. Then assess Response B's fluency. Then compare them. Only after those assessments, assign scores and choose a winner.

Score/winner consistency rules:
- If Response A is meaningfully more fluent, assign Response A a higher fluency score and set winner to "A".
- If Response B is meaningfully more fluent, assign Response B a higher fluency score and set winner to "B".
- If both responses are similarly fluent, assign the same fluency score and set winner to "tie".
- If response_a_score > response_b_score, winner must be "A".
- If response_b_score > response_a_score, winner must be "B".
- If response_a_score = response_b_score, winner must be "tie".
- Do not assign different scores for tiny differences that do not meaningfully affect fluency.

Tie type definitions:
- "both_good": both responses are fluent or fully fluent, usually scores 4–5.
- "both_bad": both responses are mostly non-fluent, garbled, or hard to read, usually scores 1–2.
- "similarly_mixed": both responses are understandable but have comparable fluency issues, usually around score 3.
- "not_a_tie": use this when winner is "A" or "B".

Return only valid JSON. Do not include markdown, commentary, or text outside the JSON.

Use exactly this JSON schema, preserving this field order:

{
  "response_a_assessment": "<1-2 short sentences explaining Response A's grammar, phrasing, flow, and any disruptive fluency issues>",
  "response_b_assessment": "<1-2 short sentences explaining Response B's grammar, phrasing, flow, and any disruptive fluency issues>",
  "comparative_reasoning": "<1 short sentence explaining which response is more fluent, or why they are tied>",
  "response_a_score": <1-5 integer>,
  "response_b_score": <1-5 integer>,
  "winner": "A" | "B" | "tie",
  "tie_type": "both_good" | "both_bad" | "similarly_mixed" | "not_a_tie"
}

[USER PROMPT]
{prompt}

[RESPONSE A]
{response_a}

[RESPONSE B]
{response_b}