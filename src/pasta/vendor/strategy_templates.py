"""
Pure functions for field derivation and prompt construction across 9 prompting strategies.

Each strategy builder returns a dict with:
  - base_prompt: str (raw text for base model)
  - aligned_prompt: str | None (raw text for aligned model in matched strategies)
  - aligned_messages: list[dict] | None (chat messages for aligned model in non-matched strategies)

Matched strategies (B1, B2, B3, F): aligned_prompt == base_prompt, aligned_messages is None
Non-matched strategies (A, C1, C2, D, E): aligned_prompt is None, aligned_messages is populated
"""

import re

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Field derivation helpers
# ---------------------------------------------------------------------------

OPENING_PHRASE_TOKENS = 50


def truncate_to_opening_phrase(text: str, max_tokens: int = OPENING_PHRASE_TOKENS) -> str:
    """Truncate text to max_tokens tokens for use as the 'Begin with' constraint."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _enc.decode(tokens[:max_tokens])
    last_space = truncated.rfind(" ")
    if last_space > len(truncated) * 0.8:
        truncated = truncated[:last_space]
    return truncated


def extract_opening_phrase(text: str, n_words: int = 10) -> str:
    """Extract first N words of text as an opening phrase."""
    words = text.split()
    return " ".join(words[:n_words])


def extract_first_sentence(text: str) -> str:
    """Extract the first complete sentence from text for 'Begin with' constraints.

    Returns the first sentence (up to ~30 words max) to give the aligned model
    a meaningful anchor without being overly long.
    """
    # Try to find a sentence-ending boundary
    match = re.search(r'[.!?]["\')\]]?\s', text)
    if match and match.end() < len(text) * 0.8:
        sentence = text[: match.end()].strip()
        # Cap at ~30 words to keep "Begin with" constraint reasonable
        words = sentence.split()
        if len(words) <= 30:
            return sentence
    # Fallback: first 15-20 words ending at a natural break (comma, semicolon)
    words = text.split()
    phrase = " ".join(words[:20])
    # Try to end at a comma or semicolon
    for sep in [",", ";", " --", " -"]:
        idx = phrase.rfind(sep)
        if idx > len(phrase) * 0.4:
            return phrase[: idx + len(sep)].strip()
    return " ".join(words[:15])


def extract_opening_sentences(text: str, n: int = 2, max_words: int = 30) -> str:
    """Extract up to N sentences from text, stopping before exceeding max_words total.

    Used for domains where the first sentence alone is too short to anchor the
    instruct model (e.g. creative fiction). Falls back to extract_first_sentence
    if no second sentence boundary is found within the word budget.
    """
    remaining = text
    collected = []
    total_words = 0
    for _ in range(n):
        match = re.search(r'[.!?]["\')\]]?\s', remaining)
        if not match or match.end() >= len(remaining) * 0.8:
            break
        sentence = remaining[: match.end()].strip()
        words = sentence.split()
        if total_words + len(words) > max_words:
            break
        collected.append(sentence)
        total_words += len(words)
        remaining = remaining[match.end():].strip()
    if collected:
        return " ".join(collected)
    return extract_first_sentence(text)


def extract_first_narrative_sentence(prefix: str) -> str:
    """For college essays: skip title, ToC, and section headers to find first prose."""
    lines = prefix.split("\n")
    header_patterns = [
        r"^table of contents",
        r"^\s*\d+\.\s",          # numbered list items like " 1. Introduction"
        r"^introduction$",
        r"^conclusion$",
        r"^abstract$",
        r"^references$",
        r"^bibliography$",
    ]
    narrative_lines = []
    skipped_first_line = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip the first line (typically the essay title)
        if not skipped_first_line:
            skipped_first_line = True
            continue
        # Skip lines matching header patterns
        if any(re.match(p, stripped, re.IGNORECASE) for p in header_patterns):
            continue
        # Skip very short lines that look like headers (< 30 chars, no sentence-ending punctuation)
        if len(stripped) < 30 and not re.search(r"[.!?]$", stripped):
            continue
        narrative_lines.append(stripped)

    if narrative_lines:
        return " ".join(narrative_lines)
    # Fallback: return everything after the first line
    rest = "\n".join(lines[1:]).strip()
    return rest if rest else prefix



def clean_wp_prompt(instruction: str) -> str:
    """Strip [WP], [TT], [CC], [EU], [RF], [PI] etc. tags from WritingPrompts instruction."""
    return re.sub(r"^\[\s*\w+\s*\]\s*", "", instruction).strip()


def extract_highlights(instruction: str) -> str:
    """Extract the highlights portion from a news instruction."""
    prefix_str = "Write a news article about the following: "
    if instruction.startswith(prefix_str):
        return instruction[len(prefix_str):]
    return instruction


def extract_essay_topic(entry: dict) -> str:
    """Extract topic from a college essay entry using the source URL slug."""
    source_url = entry.get("source_meta", {}).get("source_url", "")
    if source_url:
        slug = source_url.rstrip("/").rsplit("/", 1)[-1]
        for suffix in ("-presentation", "-coursework", "-term-paper", "-research-paper", "-essay"):
            if slug.endswith(suffix):
                slug = slug[: -len(suffix)]
                break
        return slug.replace("-", " ").title()
    # Fallback: first line of prefix with trailing "Essay" stripped
    first_line = entry["prefix"].split("\n")[0].strip()
    return re.sub(r"\s+Essay\s*$", "", first_line)


def extract_code_task(entry: dict) -> str:
    """Extract task description from code entry's docstring."""
    prefix = entry["prefix"]
    match = re.search(r'"""(.*?)"""', prefix, re.DOTALL)
    if not match:
        match = re.search(r"'''(.*?)'''", prefix, re.DOTALL)
    if match:
        return match.group(1).strip()
    return entry.get("instruction", prefix).strip()


def extract_science_topic(entry: dict) -> str:
    """Extract paper title from source_meta, falling back to first sentence of prefix."""
    title = entry.get("source_meta", {}).get("title", "")
    if title:
        return title
    # Fallback for older prefix files without title in source_meta
    cleaned = entry["prefix"].replace("\n", " ").strip()
    first_sentence = cleaned.split(".")[0].strip()
    if len(first_sentence) > 150:
        first_sentence = first_sentence[:150]
    return first_sentence


def extract_opinion_topic(entry: dict) -> str:
    """Extract topic from an opinion piece entry using the title from source_meta."""
    title = entry.get("source_meta", {}).get("title", "")
    if title:
        return title
    cleaned = entry["prefix"].replace("\n", " ").strip()
    first_sentence = cleaned.split(".")[0].strip()
    return first_sentence[:150] if len(first_sentence) > 150 else first_sentence


def derive_fields(domain: str, entry: dict) -> dict:
    """Derive normalized fields from a prefix entry for use in strategy templates.

    Returns dict with: topic, seed_opening, opening_phrase, prefix, instruction,
    human_continuation, and domain-specific extras.
    """
    fields = {
        "prefix": entry["prefix"],
        "instruction": entry["instruction"],
        "human_continuation": entry["human_continuation"],
        "source_meta": entry.get("source_meta", {}),
    }

    if domain == "college_essays":
        fields["topic"] = extract_essay_topic(entry)
        fields["seed_opening"] = extract_first_narrative_sentence(entry["prefix"])
        fields["opening_phrase"] = truncate_to_opening_phrase(entry["prefix"])

    elif domain == "news_articles":
        fields["highlights"] = extract_highlights(entry["instruction"])
        fields["opening_phrase"] = truncate_to_opening_phrase(entry["prefix"])

    elif domain == "scientific_abstracts":
        fields["topic"] = extract_science_topic(entry)
        fields["seed_opening"] = entry["prefix"]
        fields["opening_phrase"] = truncate_to_opening_phrase(entry["prefix"])

    elif domain == "creative_fiction":
        fields["clean_prompt"] = clean_wp_prompt(entry["instruction"])
        fields["topic"] = fields["clean_prompt"]
        fields["opening_phrase"] = truncate_to_opening_phrase(entry["prefix"])

    elif domain == "code_python":
        fields["task"] = extract_code_task(entry)
        fields["entry_point"] = entry.get("source_meta", {}).get("entry_point", "")
        fields["seed_code"] = entry["prefix"]
        fields["topic"] = fields["task"]
        fields["seed_opening"] = entry["prefix"]
        fields["opening_phrase"] = ""  # Not applicable for code

    elif domain == "opinion_pieces":
        fields["topic"] = extract_opinion_topic(entry)
        fields["seed_opening"] = extract_first_narrative_sentence(entry["prefix"])
        fields["opening_phrase"] = truncate_to_opening_phrase(entry["prefix"])

    return fields


# ---------------------------------------------------------------------------
# Continuation and exemplar text builders
# ---------------------------------------------------------------------------

def build_continuation_text(prefix: str, human_continuation: str, domain: str, target_words: int = 150) -> str:
    """Build a human-written passage for Strategy F by combining prefix + beginning of continuation."""
    if domain == "code_python":
        # For code: prefix + first few lines of canonical solution
        lines = human_continuation.strip().split("\n")
        # Take first 3-5 non-empty lines
        code_lines = [l for l in lines if l.strip()][:5]
        return prefix + "\n".join(code_lines) + "\n"

    combined = prefix + " " + human_continuation
    words = combined.split()
    if len(words) <= target_words:
        return combined.strip()

    # Truncate at sentence boundary near target
    truncated = " ".join(words[:target_words])
    # Find last sentence-ending punctuation
    matches = list(re.finditer(r'[.!?]["\')\]]?\s', truncated))
    if matches:
        last_match = matches[-1]
        return truncated[: last_match.end()].strip()
    return truncated.strip()


def build_exemplar_text(entry: dict, domain: str, max_words: int = 250) -> str:
    """Build exemplar text for Strategy C1 from a prefix entry."""
    if domain == "code_python":
        # For code: full prefix + full canonical solution
        return entry["prefix"] + entry["human_continuation"]

    combined = entry["prefix"] + " " + entry["human_continuation"]
    words = combined.split()
    if len(words) <= max_words:
        return combined.strip()

    truncated = " ".join(words[:max_words])
    # Truncate at sentence boundary
    matches = list(re.finditer(r'[.!?]["\')\]]?\s', truncated))
    if matches:
        last_match = matches[-1]
        return truncated[: last_match.end()].strip()
    return truncated.strip()


# ---------------------------------------------------------------------------
# Domain labels and persona definitions
# ---------------------------------------------------------------------------

DOMAIN_LABELS = {
    "college_essays": ("academic essays", "Essay"),
    "news_articles": ("news articles", "Article"),
    "scientific_abstracts": ("scientific abstracts", "Abstract"),
    "creative_fiction": ("short stories", "Story"),
    "code_python": ("Python implementations", "Implementation"),
    "opinion_pieces": ("opinion pieces", "Opinion Piece"),
}

EXEMPLAR_LABELS = {
    "college_essays": "academic essay",
    "news_articles": "news article",
    "scientific_abstracts": "scientific abstract",
    "creative_fiction": "short story",
    "code_python": "Python implementation",
    "opinion_pieces": "opinion piece",
}

PERSONA_BASE = {
    "college_essays": "The following essay was written by a college student for a university course assignment.",
    "news_articles": "The following article was written by a news correspondent.",
    "scientific_abstracts": "The following abstract is by a researcher, published in a peer-reviewed journal.",
    "creative_fiction": "The following is a story written by an aspiring fiction writer.",
    "code_python": "The following code was written by an experienced software engineer.",
    "opinion_pieces": "The following opinion piece was written by a writer for an online publication.",
}

PERSONA_SYSTEM = {
    "college_essays": "You are a college student writing an essay for a university course. Write in a clear academic style.",
    "news_articles": "You are a journalist writing a news article. Write in standard journalistic style.",
    "scientific_abstracts": "You are a researcher writing a scientific abstract. Write in precise academic style.",
    "creative_fiction": "You are a fiction writer. Write in a polished narrative style.",
    "code_python": "You are an experienced software engineer. Write clean, efficient, well-documented Python code. Output only code.",
    "opinion_pieces": "You are a writer composing an opinion piece for a general audience. Write persuasively with a clear thesis and supporting arguments.",
}


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def build_strategy_A(domain: str, fields: dict) -> dict:
    """Strategy A: Naturalistic — each model prompted in its most natural format."""
    # Base: document-style framing + prefix
    if domain == "college_essays":
        base_prompt = f"The following is an academic essay.\n\n{fields['prefix']}"
        aligned_messages = [{"role": "user", "content": f"Write an academic essay about: {fields['topic']}\n\nBegin with: \"{fields['opening_phrase']}\""}]
    elif domain == "news_articles":
        base_prompt = fields["prefix"]  # Self-framing with CNN markers
        aligned_messages = [{"role": "user", "content": f"{fields['instruction']}\n\nBegin with: \"{fields['opening_phrase']}\""}]
    elif domain == "scientific_abstracts":
        base_prompt = f"Abstract:\n{fields['prefix']}"
        aligned_messages = [{"role": "user", "content": f"{fields['instruction']}\n\nBegin with: \"{fields['opening_phrase']}\""}]
    elif domain == "creative_fiction":
        base_prompt = f"The following is a short story.\n\n{fields['prefix']}"
        aligned_messages = [{"role": "user", "content": f"Write a short story based on this prompt: {fields['clean_prompt']}\n\nBegin with: \"{fields['opening_phrase']}\""}]
    elif domain == "code_python":
        base_prompt = fields["prefix"]  # HumanEval is self-contained
        aligned_messages = [{"role": "user", "content": f"{fields['instruction']}\n\nOutput only code, no explanations."}]
    elif domain == "opinion_pieces":
        base_prompt = f"The following is an opinion piece.\n\n{fields['prefix']}"
        aligned_messages = [{"role": "user", "content": f"Write an opinion piece about: {fields['topic']}\n\nBegin with: \"{fields['opening_phrase']}\""}]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return {"base_prompt": base_prompt, "aligned_prompt": None, "aligned_messages": aligned_messages}


def build_strategy_B1(domain: str, fields: dict) -> dict:
    """Strategy B1: Matched Completion — both models get identical completion-style prompt."""
    # Same as Strategy A base prompt
    result = build_strategy_A(domain, fields)
    prompt = result["base_prompt"]
    return {"base_prompt": prompt, "aligned_prompt": prompt, "aligned_messages": None}


def build_strategy_B2(domain: str, fields: dict) -> dict:
    """Strategy B2: Matched Instruction — both models get identical instruction text as raw text."""
    if domain == "college_essays":
        prompt = f"Write an academic essay about: {fields['topic']}\n\nEssay:\n{fields['seed_opening']}"
    elif domain == "news_articles":
        prompt = f"Write a news article about the following:\n{fields['highlights']}\n\nArticle:\n{fields['prefix']}"
    elif domain == "scientific_abstracts":
        prompt = f"Write a scientific abstract about: {fields['topic']}\n\nAbstract:\n{fields['prefix']}"
    elif domain == "creative_fiction":
        prompt = f"Write a short story based on this prompt: {fields['clean_prompt']}\n\n{fields['prefix']}"
    elif domain == "code_python":
        prompt = f"Write a Python implementation for: {fields['task']}\n\n{fields['seed_code']}"
    elif domain == "opinion_pieces":
        prompt = f"Write an opinion piece about: {fields['topic']}\n\nOpinion Piece:\n{fields['prefix']}"
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return {"base_prompt": prompt, "aligned_prompt": prompt, "aligned_messages": None}


def build_strategy_B3(domain: str, fields: dict) -> dict:
    """Strategy B3: Minimal Seed — bare seed text, no framing."""
    if domain == "college_essays":
        prompt = fields["seed_opening"]
    elif domain == "news_articles":
        prompt = fields["prefix"]
    elif domain == "scientific_abstracts":
        prompt = fields["prefix"]
    elif domain == "creative_fiction":
        prompt = fields["prefix"]
    elif domain == "code_python":
        prompt = fields["prefix"]
    elif domain == "opinion_pieces":
        prompt = fields["prefix"]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return {"base_prompt": prompt, "aligned_prompt": prompt, "aligned_messages": None}


def build_strategy_C1(domain: str, fields: dict, exemplar_text: str) -> dict:
    """Strategy C1: Few-Shot k=1 — one human exemplar + prompt."""
    plural_label, singular_label = DOMAIN_LABELS[domain]

    # Base: exemplar + seed in document format
    if domain == "code_python":
        base_prompt = (
            f"The following are {plural_label}.\n\n"
            f"{singular_label} 1:\n{exemplar_text}\n\n"
            f"{singular_label} 2:\n{fields['seed_code']}"
        )
        aligned_messages = [{"role": "user", "content": f"Here is an example Python implementation:\n\n{exemplar_text}\n\nNow write a new implementation for: {fields['task']}\n\nBegin with:\n```python\n{fields['seed_code']}\n```\n\nOutput only code, no explanations."}]
    else:
        seed = fields["seed_opening"] if domain == "college_essays" else fields["prefix"]
        base_prompt = (
            f"The following are {plural_label}.\n\n"
            f"{singular_label} 1:\n{exemplar_text}\n\n"
            f"{singular_label} 2:\n{seed}"
        )
        singular_lower = singular_label.lower()
        aligned_messages = [{"role": "user", "content": f"Here is an example {singular_lower}:\n\n{exemplar_text}\n\nNow write a new {singular_lower}. Begin with: \"{fields['opening_phrase']}\""}]

    return {"base_prompt": base_prompt, "aligned_prompt": None, "aligned_messages": aligned_messages}


def build_strategy_C2(domain: str, fields: dict, exemplar_text: str) -> dict:
    """Strategy C2: Few-Shot k=1 (Full Prefix) — one human exemplar + full Strategy A prefix.

    Same exemplar as C1 but uses the full domain-framed prefix from Strategy A
    instead of a short seed. Enables clean A-vs-C2 comparison isolating the
    exemplar effect with prefix length held constant.
    """
    plural_label, singular_label = DOMAIN_LABELS[domain]

    # Build the Strategy A base prompt for the second item
    a_result = build_strategy_A(domain, fields)
    a_base_prompt = a_result["base_prompt"]

    # Base: exemplar + full Strategy A prefix in document format
    base_prompt = (
        f"The following are {plural_label}.\n\n"
        f"{singular_label} 1:\n{exemplar_text}\n\n"
        f"{singular_label} 2:\n{a_base_prompt}"
    )

    # Aligned: exemplar + Strategy A's aligned message (prepended with exemplar intro)
    exemplar_label = EXEMPLAR_LABELS[domain]
    a_aligned_msg = a_result["aligned_messages"][-1]["content"]
    aligned_messages = [{"role": "user", "content": f"Here is an example {exemplar_label}:\n\n{exemplar_text}\n\n{a_aligned_msg}"}]

    return {"base_prompt": base_prompt, "aligned_prompt": None, "aligned_messages": aligned_messages}


def build_strategy_D(domain: str, fields: dict) -> dict:
    """Strategy D: Persona — persona framing for both models."""
    persona_desc = PERSONA_BASE[domain]
    system_msg = PERSONA_SYSTEM[domain]

    # Base: persona description + prefix
    if domain == "code_python":
        base_prompt = f"{persona_desc}\n\n# Task: {fields['task']}\n{fields['seed_code']}"
        aligned_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Implement the following: {fields['task']}\n\nBegin with:\n```python\n{fields['seed_code']}\n```"},
        ]
    elif domain == "college_essays":
        base_prompt = f"{persona_desc}\n\n{fields['prefix']}"
        aligned_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Write an essay about: {fields['topic']}\n\nBegin with: \"{fields['opening_phrase']}\""},
        ]
    elif domain == "news_articles":
        base_prompt = f"{persona_desc}\n\n{fields['prefix']}"
        aligned_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{fields['instruction']}\n\nBegin with: \"{fields['opening_phrase']}\""},
        ]
    elif domain == "scientific_abstracts":
        base_prompt = f"{persona_desc}\n\nAbstract: {fields['prefix']}"
        aligned_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{fields['instruction']}\n\nBegin with: \"{fields['opening_phrase']}\""},
        ]
    elif domain == "creative_fiction":
        base_prompt = f"{persona_desc}\n\n{fields['prefix']}"
        aligned_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Write a short story based on this prompt: {fields['clean_prompt']}\n\nBegin with: \"{fields['opening_phrase']}\""},
        ]
    elif domain == "opinion_pieces":
        base_prompt = f"{persona_desc}\n\n{fields['prefix']}"
        aligned_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Write an opinion piece about: {fields['topic']}\n\nBegin with: \"{fields['opening_phrase']}\""},
        ]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return {"base_prompt": base_prompt, "aligned_prompt": None, "aligned_messages": aligned_messages}


def build_strategy_E(domain: str, fields: dict) -> dict:
    """Strategy E: Specificity Detailed — heavily constrained prompts."""
    if domain == "college_essays":
        base_prompt = (
            f"The following is a 500-word academic essay about: {fields['topic']}. "
            "It uses a formal academic style, clear thesis statements, specific examples, "
            "and follows a structured argument with introduction, body paragraphs, and conclusion.\n\n"
            f"{fields['seed_opening']}"
        )
        aligned_messages = [{"role": "user", "content": (
            f"Write a 500-word academic essay about: {fields['topic']}\n\n"
            "Requirements:\n"
            "- Formal academic style\n"
            "- Clear thesis statements\n"
            "- Specific examples and evidence\n"
            "- Structured argument: introduction, body paragraphs, conclusion\n\n"
            f"Begin with: \"{fields['opening_phrase']}\""
        )}]
    elif domain == "news_articles":
        base_prompt = (
            "The following is a 400-word news article. It uses inverted pyramid structure, "
            "attributes claims to sources, maintains an objective tone, and includes direct quotes.\n\n"
            f"{fields['prefix']}"
        )
        aligned_messages = [{"role": "user", "content": (
            f"Write a 400-word news article about the following:\n{fields['highlights']}\n\n"
            "Requirements:\n"
            "- Inverted pyramid structure\n"
            "- Attribute all claims to sources\n"
            "- Objective tone\n"
            "- Include at least two direct quotes\n\n"
            f"Begin with: \"{fields['opening_phrase']}\""
        )}]
    elif domain == "scientific_abstracts":
        base_prompt = (
            "The following is a 250-word abstract for a research paper. "
            "It follows the standard structure: motivation, gap, method, experiments, results, and conclusion. "
            "It is written in formal academic English.\n\n"
            f"{fields['prefix']}"
        )
        aligned_messages = [{"role": "user", "content": (
            f"Write a 250-word abstract for a research paper about: {fields['topic']}.\n\n"
            "Requirements:\n"
            "- Follow standard structure: motivation, gap, method, experiments, results, conclusion\n"
            "- Formal academic English\n"
            "- Be specific about methods and quantitative results\n\n"
            f"Begin with: \"{fields['opening_phrase']}\""
        )}]
    elif domain == "creative_fiction":
        base_prompt = (
            "The following is a 500-word short story excerpt. It uses vivid sensory details, "
            "includes dialogue, maintains a consistent narrative voice, and builds tension.\n\n"
            f"{fields['prefix']}"
        )
        aligned_messages = [{"role": "user", "content": (
            f"Write a 500-word short story based on this prompt: {fields['clean_prompt']}\n\n"
            "Requirements:\n"
            "- Vivid sensory details\n"
            "- Include dialogue\n"
            "- Consistent narrative voice\n"
            "- Build tension\n"
            "- Ambiguous or open ending\n\n"
            f"Begin with: \"{fields['opening_phrase']}\""
        )}]
    elif domain == "code_python":
        base_prompt = (
            f"# Clean, efficient Python implementation of: {fields['task']}\n"
            "# Requirements: handle edge cases, include type hints, comprehensive docstrings.\n\n"
            f"{fields['seed_code']}"
        )
        aligned_messages = [{"role": "user", "content": (
            f"Write a Python implementation of: {fields['task']}\n\n"
            "Requirements:\n"
            "- Optimal time complexity\n"
            "- Handle edge cases (empty input, single element, boundary values)\n"
            "- Include type hints\n"
            "- Comprehensive docstrings\n"
            "- Output only code, no explanations\n\n"
            f"Begin with:\n```python\n{fields['seed_code']}\n```"
        )}]
    elif domain == "opinion_pieces":
        base_prompt = (
            "The following is a 500-word opinion piece. It presents a clear thesis, "
            "uses persuasive rhetoric, includes specific examples and evidence, "
            "maintains a consistent argumentative voice, and addresses potential counterarguments.\n\n"
            f"{fields['prefix']}"
        )
        aligned_messages = [{"role": "user", "content": (
            f"Write a 500-word opinion piece about: {fields['topic']}\n\n"
            "Requirements:\n"
            "- Clear thesis statement\n"
            "- Persuasive rhetoric\n"
            "- Specific examples and evidence\n"
            "- Consistent argumentative voice\n"
            "- Address potential counterarguments\n\n"
            f"Begin with: \"{fields['opening_phrase']}\""
        )}]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return {"base_prompt": base_prompt, "aligned_prompt": None, "aligned_messages": aligned_messages}


def build_strategy_F(domain: str, fields: dict) -> dict:
    """Strategy F: Continuation — both models continue from human-written passage."""
    continuation_text = build_continuation_text(fields["prefix"], fields["human_continuation"], domain, target_words=150)
    return {"base_prompt": continuation_text, "aligned_prompt": continuation_text, "aligned_messages": None}


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "A": {
        "id": "A",
        "name": "naturalistic",
        "full_name": "strategy_A_naturalistic",
        "use_chat_template_for_aligned": True,
        "description": "Each model prompted in its most natural format. Base gets document-style prefix; aligned gets chat instruction.",
        "builder": build_strategy_A,
        "needs_exemplar": False,
    },
    "B1": {
        "id": "B1",
        "name": "matched_completion",
        "full_name": "strategy_B1_matched_completion",
        "use_chat_template_for_aligned": False,
        "description": "Both models receive identical completion-style (base-native) prompt. No chat template for either.",
        "builder": build_strategy_B1,
        "needs_exemplar": False,
    },
    "B2": {
        "id": "B2",
        "name": "matched_instruction",
        "full_name": "strategy_B2_matched_instruction",
        "use_chat_template_for_aligned": False,
        "description": "Both models receive identical instruction text as raw text. No chat template for either.",
        "builder": build_strategy_B2,
        "needs_exemplar": False,
    },
    "B3": {
        "id": "B3",
        "name": "minimal_seed",
        "full_name": "strategy_B3_minimal_seed",
        "use_chat_template_for_aligned": False,
        "description": "Both models receive bare seed text only. No framing, no instructions. Exposes raw model priors.",
        "builder": build_strategy_B3,
        "needs_exemplar": False,
    },
    "C1": {
        "id": "C1",
        "name": "fewshot_1_short",
        "full_name": "strategy_C1_fewshot_1_short",
        "use_chat_template_for_aligned": True,
        "description": "One real human-written exemplar + short seed. Tests exemplar grounding with minimal anchor (compare vs B3).",
        "builder": build_strategy_C1,
        "needs_exemplar": True,
    },
    "C2": {
        "id": "C2",
        "name": "fewshot_1_full",
        "full_name": "strategy_C2_fewshot_1_full",
        "use_chat_template_for_aligned": True,
        "description": "One real human-written exemplar + full Strategy A prefix. Tests exemplar grounding with full anchor (compare vs A).",
        "builder": build_strategy_C2,
        "needs_exemplar": True,
    },
    "D": {
        "id": "D",
        "name": "persona",
        "full_name": "strategy_D_persona",
        "use_chat_template_for_aligned": True,
        "description": "Persona framing. Base gets persona in document frame; aligned gets persona in system message.",
        "builder": build_strategy_D,
        "needs_exemplar": False,
    },
    "E": {
        "id": "E",
        "name": "specificity_detailed",
        "full_name": "strategy_E_specificity_detailed",
        "use_chat_template_for_aligned": True,
        "description": "Heavily constrained prompt specifying style, length, structure, and tone.",
        "builder": build_strategy_E,
        "needs_exemplar": False,
    },
    "F": {
        "id": "F",
        "name": "continuation",
        "full_name": "strategy_F_continuation",
        "use_chat_template_for_aligned": False,
        "description": "Human-written passage (~150 words). Both models continue from same text. No chat template for either.",
        "builder": build_strategy_F,
        "needs_exemplar": False,
    },
}
