You are comparing two model responses only on PROMPT RELEVANCE.

Prompt relevance means whether each response satisfies the user's requested task. The prompt may contain up to three relevant components:

(1) TOPIC — Is the response about what the prompt asks about?
(2) GENRE/FORM — Does the response use the requested type or format, such as academic essay, short story, opinion piece, news article, list, email, code, etc.?
(3) EXPLICIT INSTRUCTIONS — Does the response honor specific constraints, such as length, structure, required sections, tone, or any "Begin with: ..." requirement?

Your task is to decide which response is more relevant to the user prompt.

Rate each response on this 1–5 relevance scale:

1 — Off-topic or non-answer.
The response addresses a different task, ignores the prompt, is generic/evasive, or does not meaningfully attempt to answer.

2 — Mostly off-topic.
The response has only a tangential connection to the prompt, uses the wrong genre/form in a way that prevents task completion, or misses most required elements.

3 — Partially relevant.
The response is on the right topic but misses major parts of the request, substantially drifts, uses the wrong form, or only loosely addresses the user's intent.

4 — Relevant.
The response addresses the main topic, intent, and requested form, with only minor gaps or omissions.

5 — Fully relevant.
The response directly and completely addresses the topic, intent, requested form, and explicit constraints.

What should make a response more relevant:
- It directly addresses the requested topic.
- It follows the requested genre/form.
- It satisfies explicit instructions and constraints.
- It answers all parts of a multi-part prompt.
- It avoids irrelevant drift.
- It completes the requested task rather than refusing, evading, or giving generic boilerplate.

What should make a response less relevant:
- It is off-topic or mostly off-topic.
- It uses the wrong genre/form.
- It ignores explicit constraints, including required openings such as "Begin with: ...".
- It answers only part of the prompt.
- It gives generic safety/policy language or refuses a benign request.
- It provides mostly background/setup without completing the requested task.
- It merely echoes or restates the prompt without substantively answering it.

Focus only on relevance. Do NOT judge:
- Fluency.
- Grammar.
- Writing quality.
- Elegance.
- Creativity.
- Persuasiveness.
- Verbosity.
- Confidence.
- Formatting polish.
- Length, unless the prompt explicitly specifies length.
- Factual accuracy, unless the prompt explicitly requires real, cited, current, verifiable, or evidence-based information.

Important clarification about factuality:
Do not penalize ordinary factual errors when judging relevance, because factuality is evaluated separately. However, if the prompt explicitly asks for real, cited, current, verifiable, or evidence-based information, and a response does not attempt to provide that type of answer, reduce its relevance because it failed an explicit requirement.

Important clarification about truncation:
Judge only the visible content. Do not speculate about what either response might have said after truncation. Do not penalize abrupt ending by itself. However, if the visible content fails to satisfy required parts of the prompt, assign a lower relevance score accordingly.

Important clarification about noisy prompts:
If the prompt contains boilerplate, copied context, or noisy text, identify the user's evident task. However, do not ignore explicit constraints unless they are clearly unrelated boilerplate or contradictory with the main task.

Important clarification about ties:
Use "tie" when both responses are similarly relevant or similarly irrelevant.
Do not force a winner based on fluency, style, length, formatting, factuality, confidence, or polish.
If both responses fully satisfy the prompt, choose "tie".
If both responses fail the prompt in similar ways, choose "tie".

First assess Response A's relevance. Then assess Response B's relevance. Then compare them. Only after those assessments, assign scores and choose a winner.

Score/winner consistency rules:
- If Response A is meaningfully more relevant, assign Response A a higher relevance score and set winner to "A".
- If Response B is meaningfully more relevant, assign Response B a higher relevance score and set winner to "B".
- If both responses are similarly relevant, assign the same relevance score and set winner to "tie".
- If response_a_score > response_b_score, winner must be "A".
- If response_b_score > response_a_score, winner must be "B".
- If response_a_score = response_b_score, winner must be "tie".
- Do not assign different scores for tiny differences that do not meaningfully affect prompt relevance.

Tie type definitions:
- "both_good": both responses are relevant or fully relevant, usually scores 4–5.
- "both_bad": both responses are mostly irrelevant, off-topic, or non-answers, usually scores 1–2.
- "similarly_mixed": both responses are partially relevant or have comparable relevance issues, usually around score 3.
- "not_a_tie": use this when winner is "A" or "B".

Return only valid JSON. Do not include markdown, commentary, or text outside the JSON.

Use exactly this JSON schema, preserving this field order:

{
  "response_a_assessment": "<1-2 short sentences explaining how Response A addresses or misses the prompt topic, genre/form, and explicit instructions>",
  "response_b_assessment": "<1-2 short sentences explaining how Response B addresses or misses the prompt topic, genre/form, and explicit instructions>",
  "comparative_reasoning": "<1 short sentence explaining which response is more relevant, or why they are tied>",
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