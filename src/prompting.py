"""Prompt construction for the generator, kept free of heavy imports so it can be unit-tested."""

INSTRUCTION = "Answer the question based on the context below."


def _ids(tokenizer, text, max_length, special=False):
    """Token ids of ``text``, truncated to ``max_length`` (never tokenises past the limit)."""
    return tokenizer(text, add_special_tokens=special, truncation=True, max_length=max_length)["input_ids"]


def _assemble(query, passages):
    context = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"{INSTRUCTION}\nContext: {context}\nQuestion: {query}\nAnswer:"


def build_prompt(tokenizer, query, docs, max_input_tokens):
    """Build a prompt that fits in ``max_input_tokens`` and contains every document.

    Each document is truncated to the same token budget, so all retrieved
    documents reach the generator instead of the first one filling the window.
    Returns ``(prompt, n_tokens)`` where ``n_tokens`` includes special tokens and
    is guaranteed to be <= ``max_input_tokens``.
    """
    n_docs = max(len(docs), 1)
    overhead = len(_ids(tokenizer, _assemble(query, [""] * len(docs)), max_input_tokens, special=True))
    budget = max((max_input_tokens - overhead) // n_docs, 1)
    while True:
        passages = [
            tokenizer.decode(_ids(tokenizer, doc, budget), skip_special_tokens=True) for doc in docs
        ]
        prompt = _assemble(query, passages)
        # Count with a cap of limit + 1: longer than the limit <=> the cap is reached.
        n_tokens = len(_ids(tokenizer, prompt, max_input_tokens + 1, special=True))
        if n_tokens <= max_input_tokens or budget <= 1:
            break
        budget -= max(1, (n_tokens - max_input_tokens + n_docs - 1) // n_docs)
        budget = max(budget, 1)
    if n_tokens > max_input_tokens:
        raise ValueError(
            f"Prompt needs {n_tokens} tokens but the limit is {max_input_tokens}; "
            "use fewer documents or a shorter query."
        )
    return prompt, n_tokens
