"""Conversations as training tokens, for bridge batches and for supervised fine-tuning.

`encode_conversation` renders a conversation once with the policy's chat template, tokenizes the
text once, and marks the assistant turns as trainable. A turn's span is the text that rendering
one more message adds after the generation prompt, so the mask follows the template exactly:
Qwen3.5/3.6 fold consecutive tool results into one turn and keep reasoning only after the last
user query, which rendering messages one at a time (verl's `MultiTurnSFTDataset`) gets wrong.

Nothing here imports verl, and torch only when a dataset item is built.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoenvscaling.trajectories.export import read_jsonl

Messages = list[dict[str, Any]]


def encode_conversation(messages: Messages, tools: list[dict] | None, tokenizer: Any,
                        **template_kwargs: Any) -> tuple[list[int], list[int]]:
    """Token ids of the rendered conversation and a mask that is 1 on assistant turns."""
    def render(upto: int, prompt: bool) -> str:
        return tokenizer.apply_chat_template(messages[:upto], tools=tools or None, tokenize=False,
                                             add_generation_prompt=prompt, **template_kwargs)

    text = render(len(messages), False)
    spans = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start, turn = render(i, True), render(i + 1, False)
        if not (turn.startswith(start) and text.startswith(turn)):
            raise ValueError(f"the chat template does not render message {i} as a prefix of the "
                             "conversation; its assistant turn cannot be located")
        # The turn ends at its stop token: the newline the template writes after it is not generated.
        spans.append((len(start), len(turn.rstrip())))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [lo for lo, _ in spans]

    def trained(begin: int) -> int:
        i = bisect_right(starts, begin) - 1
        return int(i >= 0 and begin < spans[i][1])

    return list(encoded["input_ids"]), [trained(begin) for begin, _ in encoded["offset_mapping"]]


def split_sample(ids: Sequence[int], mask: Sequence[int],
                 max_tokens: int) -> tuple[list[int], list[int], list[int]]:
    """(prompt, response, response mask): the response starts at the first trainable token.

    A sample longer than `max_tokens`, or with nothing to train on, is cut and made inert (an
    all-zero mask, Miles' `remove_sample`): it stays in its group without carrying gradient.
    """
    inert = len(ids) > max_tokens or 1 not in mask
    ids, mask = list(ids[:max_tokens]), list(mask[:max_tokens])
    cut = max(1, len(ids) - 1 if inert else mask.index(1))
    return ids[:cut], ids[cut:], [0] * (len(ids) - cut) if inert else mask[cut:]


def rows_by_group(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Bridge rows (`trajectories.export.rl_sample_rows`) by group, members in index order."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["group_index"])].append(row)
    return {g: sorted(members, key=lambda r: int(r["index"])) for g, members in sorted(groups.items())}


class ConversationDataset:
    """verl's SFT dataset (`data.custom_cls`) over `export sft` rows: messages and tools.

    Reads the JSONL rows as written: a Parquet round trip would merge every tool call's
    arguments into one Arrow struct and hand the template keys the model never produced.
    Items are what verl's `no_padding` mode expects: input_ids, position_ids, loss_mask.
    """

    def __init__(self, parquet_files: str | Sequence[str], tokenizer: Any, config: Any,
                 processor: Any = None, max_samples: int = -1):
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [row for path in paths for row in read_jsonl(Path(path))]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.tokenizer = tokenizer
        self.messages_key = config.get("messages_key", "messages")
        self.tools_key = config.get("tools_key", "tools")
        self.max_length = int(config.get("max_length", 1 << 30))
        self.truncation = config.get("truncation", "right")
        self.template_kwargs = dict(config.get("apply_chat_template_kwargs") or {})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch

        row = self.rows[item]
        ids, mask = encode_conversation(row[self.messages_key], row.get(self.tools_key), self.tokenizer,
                                        **self.template_kwargs)
        if len(ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"row {item}: {len(ids)} tokens exceed max_length={self.max_length}")
            keep = slice(-self.max_length, None) if self.truncation == "left" else slice(self.max_length)
            ids, mask = ids[keep], mask[keep]
        return {"input_ids": torch.tensor(ids), "position_ids": torch.arange(len(ids)),
                "loss_mask": torch.tensor(mask)}
