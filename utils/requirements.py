import re
from typing import Dict, Any


_EXPERIMENTAL = re.compile(
    r"\b(experiment|experimental|evidence|result|metric|bleu|accuracy|"
    r"table\s*\d|ablation|evaluation|benchmark|dataset|score|training\s+cost)\b",
    re.I,
)
_ARCHITECTURE = re.compile(
    r"\b(architectur|component|encoder|decoder|attention|layer|residual|"
    r"masking|positional|multi-head|self-attention|feed-forward)\b",
    re.I,
)
_THEORETICAL = re.compile(
    r"\b(theor|theoretical|complexity|asymptotic|path\s+length|o\s*\(|"
    r"compare.*rnn|compare.*cnn|parallelization|dependency)\b",
    re.I,
)


def detect_requirements(query: str) -> Dict[str, Any]:
    """Rule-based flags aligned with common research question phrasing."""
    return {
        "requires_experimental": bool(_EXPERIMENTAL.search(query)),
        "requires_architecture": bool(_ARCHITECTURE.search(query)),
        "requires_theoretical": bool(_THEORETICAL.search(query)),
        "requires_any_evidence": bool(
            _EXPERIMENTAL.search(query)
            or _ARCHITECTURE.search(query)
            or _THEORETICAL.search(query)
        ),
    }


def format_requirements_for_prompt(requirements: Dict[str, Any]) -> str:
    lines = []
    if requirements.get("requires_experimental"):
        lines.append(
            "- EXPERIMENTAL: notes must include specific numbers, metrics, and/or Table/Figure "
            "references when present in sources (e.g. BLEU scores, training cost)."
        )
    if requirements.get("requires_architecture"):
        lines.append(
            "- ARCHITECTURAL: notes must name concrete components (e.g. multi-head attention, "
            "residual connections, encoder-decoder attention, masking, positional encoding)."
        )
    if requirements.get("requires_theoretical"):
        lines.append(
            "- THEORETICAL: notes must include explicit comparisons (e.g. path length O(1) vs "
            "O(n), parallelization, long-range dependencies) when in sources."
        )
    if not lines:
        lines.append(
            "- GENERAL: notes must address all parts of the question with source-backed bullets."
        )
    return "\n".join(lines)
