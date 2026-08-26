"""Robust chat-template rendering shared by surgery and evaluation paths."""

from __future__ import annotations

from typing import Any


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Render one user prompt with a compatible text-content representation.

    Most chat templates expect ``message["content"]`` to be a string. Safety
    classifiers such as Llama Guard instead expect OpenAI-style typed content
    blocks. Try both representations and require the rendered prompt to retain
    the source text so a permissive Jinja template cannot silently emit an empty
    conversation.
    """

    conversations = (
        [{"role": "user", "content": prompt}],
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )
    errors: list[BaseException] = []
    for conversation in conversations:
        try:
            try:
                rendered = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as error:
            errors.append(error)
            continue
        if not isinstance(rendered, str):
            errors.append(TypeError("chat template did not return text"))
            continue
        if prompt not in rendered:
            errors.append(ValueError("chat template omitted the source prompt"))
            continue
        return rendered

    if errors:
        raise ValueError("no compatible chat-template text representation") from errors[-1]
    raise ValueError("no compatible chat-template text representation")
