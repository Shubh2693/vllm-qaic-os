#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Filler prose used to pad a prompt to an exact token length.

Each paragraph is tagged with a running section number when tiled. Perfectly periodic
filler is itself a strong inducement to degenerate output, which would make
``ctx_assertions.assert_not_degenerate`` fail for reasons that have nothing to do with
the accelerator; the section markers break the periodicity cheaply.
"""

_FILLER_PARAGRAPHS = (
    "Artificial intelligence has moved from a specialised branch of computer science "
    "into a foundational technology that touches transport, medicine, logistics and "
    "the everyday operation of consumer devices. Its progress has been uneven, marked "
    "by long periods of reduced funding followed by abrupt bursts of capability.",
    "Early systems encoded human expertise as explicit rules. They could play board "
    "games and answer narrow questions, but they were brittle: every new situation "
    "demanded another rule, and ambiguity in the real world defeated them. The limits "
    "of that approach shaped a generation of research priorities.",
    "Learning from data rather than from hand-written rules changed the trajectory. "
    "Layered networks extracted their own hierarchies of features from raw pixels, "
    "waveforms and characters, and tasks that had resisted decades of manual "
    "engineering began to yield to scale and gradient descent.",
    "Three ingredients drove the shift: an abundance of digitised data, hardware that "
    "made large matrix products cheap, and training recipes stable enough to be "
    "reproduced by others. Remove any one and the progress of the last decade looks "
    "very different.",
    "In medicine, models now assist with triage, image review and the prediction of "
    "patient trajectories. The gains are real but conditional -- they depend on how "
    "closely deployment data resembles training data, and on clinicians retaining the "
    "authority and the information needed to overrule a recommendation.",
    "Language models compress an enormous amount of text into weights that can be "
    "queried conversationally. They summarise, translate, draft and refactor, and they "
    "fail in characteristic ways: confident invention of detail, sensitivity to the "
    "phrasing of a request, and uneven performance across languages.",
    "Long context changed how these systems are used. A model that can attend across "
    "hundreds of thousands of tokens can read an entire codebase, a lengthy contract, "
    "or hours of transcribed video in one pass, and answer questions that require "
    "joining facts separated by great distances in the input.",
    "Serving such models efficiently is mostly a memory problem. The cache of keys and "
    "values grows linearly with sequence length and with the number of layers, so the "
    "arithmetic of a long-context deployment is dominated by how much of that cache "
    "fits in fast memory close to the compute.",
    "Multimodal inputs complicate the picture further. Images and video are converted "
    "into sequences of visual tokens whose count depends on resolution and on frame "
    "rate, which means the same nominal context window holds very different amounts of "
    "content depending on what is placed in it.",
    "Evaluation has struggled to keep pace with capability. Benchmarks saturate, leak "
    "into training data, and reward narrow forms of competence. Measuring whether a "
    "system behaves sensibly at the edges of its declared limits requires tests built "
    "specifically for those edges.",
)


def _tile_filler_ids(tokenizer, min_tokens: int, start_section: int = 0) -> list[int]:
    """Tokenise numbered filler paragraphs until at least min_tokens ids exist.

    ``start_section`` rotates both the paragraph order and the section numbering, which
    is how one batch of requests gets prose that differs per request at the same
    length. Without it the requests would be byte-identical and any prefix reuse would
    make the batch cheaper than the case claims.
    """
    ids: list[int] = []
    section = start_section
    while len(ids) < min_tokens:
        paragraph = _FILLER_PARAGRAPHS[section % len(_FILLER_PARAGRAPHS)]
        chunk = f"\n\nSection {section + 1}. {paragraph}"
        ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        section += 1
    return ids


def text_of_exact_len(tokenizer, num_tokens: int, variant: int = 0) -> str:
    """Build text that re-encodes to exactly num_tokens ids where possible.

    Slicing token ids and decoding is not round-trip exact -- the decoded string may
    re-encode to a slightly different length, and with a real BPE tokenizer the
    correction can oscillate rather than converge. So iterate, remember the closest
    candidate that does not overshoot, and fall back to it if no iteration lands
    exactly. Overshooting matters more than undershooting: too long risks blowing
    max_model_len, while too short is absorbed by TOKEN_SLACK.

    ``variant`` selects a different rotation of the filler at the same length, so a
    batch of N requests is N distinct prompts. The convergence loop absorbs the
    one-token wobble between, say, "Section 9." and "Section 10.".
    """
    if num_tokens <= 0:
        return ""

    ids = _tile_filler_ids(tokenizer, num_tokens, start_section=variant)
    text = tokenizer.decode(ids[:num_tokens], skip_special_tokens=True)

    best_text, best_gap = None, None
    for _ in range(8):
        current = tokenizer.encode(text, add_special_tokens=False)
        if len(current) == num_tokens:
            return text
        if len(current) < num_tokens:
            gap = num_tokens - len(current)
            if best_gap is None or gap < best_gap:
                best_text, best_gap = text, gap
            extra_ids = _tile_filler_ids(tokenizer, gap, start_section=variant)[:gap]
            text = text + tokenizer.decode(extra_ids, skip_special_tokens=True)
        else:
            text = tokenizer.decode(current[:num_tokens], skip_special_tokens=True)

    # No iteration landed exactly; prefer the closest under-length candidate.
    if best_text is not None:
        return best_text
    # Everything overshot -- trim hard so we never exceed the requested length.
    return tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False)[:num_tokens],
        skip_special_tokens=True,
    )
