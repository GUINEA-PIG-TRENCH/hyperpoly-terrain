"""Canonical FracType schema — shared contract between all FracType subsystems.

v2: nested vinculums, role-distinguished bars, compress-to-budget.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class Scale(Enum):
    SMALL = "small"
    NORMAL = "normal"
    LARGE = "large"


class VinculumRole(Enum):
    """Eight roles/variants of the vinculum per historical and modern theory."""
    GROUPING = "grouping"          # overline binding expressions (Chuquet 1484)
    DIVISION = "division"          # fraction bar (Fibonacci 1202)
    MULTIPLICATION = "multiplication"  # Roman numeral extension (V̄ = 5,000)
    REPETITION = "repetition"      # periodic notation (vinculum over repeating digits)
    RADICAL = "radical"            # square root / radical (Rudolff 1525)
    CONJUGATE = "conjugate"        # mirror / complex conjugate
    COMPLEMENT = "complement"      # set complement
    NEGATION = "negation"          # logical negation / strikethrough


# ── Bar glyphs per role ──
ROLE_GLYPH = {
    VinculumRole.GROUPING: "‾",        # overline U+203E
    VinculumRole.DIVISION: "─",         # horizontal box-drawing U+2500
    VinculumRole.MULTIPLICATION: "═",   # double horizontal U+2550
    VinculumRole.REPETITION: "┄",       # dashed U+2504
    VinculumRole.RADICAL: "─",          # radical horizontal extension
    VinculumRole.CONJUGATE: "‾",        # conjugate overline
    VinculumRole.COMPLEMENT: "ᶜ",       # complement suffix U+1D9C
    VinculumRole.NEGATION: "̶",          # combining long stroke overlay U+0336
}

ROLE_GLYPH_ASCII = {
    VinculumRole.GROUPING: "_",
    VinculumRole.DIVISION: "-",
    VinculumRole.MULTIPLICATION: "=",
    VinculumRole.REPETITION: "~",
    VinculumRole.RADICAL: "r",
    VinculumRole.CONJUGATE: "*",
    VinculumRole.COMPLEMENT: "c",
    VinculumRole.NEGATION: "~",
}


@dataclass
class FracNode:
    """A vinculum-compressed pair. v2: top/bottom can be str or nested FracNode."""
    top: Union[str, "FracNode"]
    bottom: Union[str, "FracNode"] = ""
    title: str = ""
    scale: Scale = Scale.NORMAL
    color_hint: str = "nominal"
    role: VinculumRole = VinculumRole.DIVISION

    # ── Flattened display strings (lazy-computed) ──
    def top_text(self) -> str:
        if isinstance(self.top, FracNode):
            return self.top.to_inline()
        return self.top

    def bottom_text(self) -> str:
        if not self.bottom:
            return ""
        if isinstance(self.bottom, FracNode):
            return self.bottom.to_inline()
        return self.bottom

    # ── Depth (0 = leaf, N = deepest nesting) ──
    def depth(self) -> int:
        d = 0
        if isinstance(self.top, FracNode):
            d = max(d, self.top.depth() + 1)
        if isinstance(self.bottom, FracNode):
            d = max(d, self.bottom.depth() + 1)
        return d

    # ── Character cost of each output format ──
    def cost_terminal(self) -> int:
        w = max(len(self.top_text()), len(self.bottom_text()), len(self.title)) + 2
        lines = 3
        if self.title:
            lines += 1
        return w * lines  # rough: worst-case char count

    def cost_inline(self) -> int:
        return len(self.to_inline())

    # ── Export ──
    def to_react(self) -> str:
        top = self.top.to_react() if isinstance(self.top, FracNode) else f'"{self.top}"'
        if self.bottom:
            bottom = self.bottom.to_react() if isinstance(self.bottom, FracNode) else f'"{self.bottom}"'
            return (
                f'<VinculumTypography top={top} bottom={bottom} '
                f'scale="{self.scale.value}" role="{self.role.value}" />'
            )
        else:
            return (
                f'<VinculumTypography top={top} '
                f'scale="{self.scale.value}" role="{self.role.value}" />'
            )

    def to_latex(self) -> str:
        top = self.top.to_latex() if isinstance(self.top, FracNode) else f"\\text{{{self.top}}}"
        if self.bottom:
            bottom = self.bottom.to_latex() if isinstance(self.bottom, FracNode) else f"\\text{{{self.bottom}}}"
            if self.role == VinculumRole.RADICAL:
                return f"\\sqrt[{bottom}]{{{top}}}"
            return f"\\frac{{{top}}}{{{bottom}}}"
        else:
            if self.role == VinculumRole.RADICAL:
                return f"\\sqrt{{{top}}}"
            elif self.role == VinculumRole.GROUPING or self.role == VinculumRole.CONJUGATE:
                return f"\\overline{{{top}}}"
            elif self.role == VinculumRole.MULTIPLICATION:
                return f"\\overline{{\\overline{{{top}}}}}"
            elif self.role == VinculumRole.REPETITION:
                return f"\\overline{{{top}}}"
            elif self.role == VinculumRole.COMPLEMENT:
                return f"{{{top}}}^{{c}}"
            elif self.role == VinculumRole.NEGATION:
                return f"\\xout{{{top}}}"
            return top

    def to_terminal_lines(self, glyph: str = None) -> list[str]:
        """Terminal vinculum block. Respects role for bar glyph and single-valued states."""
        top = self.top_text()
        bottom = self.bottom_text()
        
        # Double-valued representation
        if bottom:
            if glyph is None:
                glyph = ROLE_GLYPH.get(self.role, "─")
            width = max(len(top), len(bottom), len(self.title)) + 2
            lines = []
            if self.title:
                lines.append(f"[{self.title}]")
            
            if self.role == VinculumRole.RADICAL:
                lines.append(f" √ {top}".center(width))
                lines.append("─" * width)
                lines.append(bottom.center(width))
            else:
                lines.append(top.center(width))
                lines.append(glyph * width)
                lines.append(bottom.center(width))
            return lines

        # Single-valued representation
        width = max(len(top), len(self.title)) + 2
        lines = []
        if self.title:
            lines.append(f"[{self.title}]")
            
        if self.role == VinculumRole.DIVISION:
            lines.append(top.center(width))
            lines.append("─" * width)
            lines.append(" ".center(width))
        elif self.role == VinculumRole.GROUPING or self.role == VinculumRole.CONJUGATE:
            glyph = ROLE_GLYPH.get(self.role, "‾")
            lines.append(glyph * len(top))
            lines.append(top)
        elif self.role == VinculumRole.MULTIPLICATION:
            glyph = ROLE_GLYPH.get(self.role, "═")
            lines.append(glyph * len(top))
            lines.append(top)
        elif self.role == VinculumRole.REPETITION:
            glyph = ROLE_GLYPH.get(self.role, "┄")
            lines.append(glyph * len(top))
            lines.append(top)
        elif self.role == VinculumRole.RADICAL:
            lines.append(" ┌" + "─" * len(top))
            lines.append(f"√ {top}")
        elif self.role == VinculumRole.COMPLEMENT:
            lines.append(f"{top}ᶜ")
        elif self.role == VinculumRole.NEGATION:
            negated = "".join(ch + "̶" for ch in top)
            lines.append(negated)
        else:
            lines.append(top)
            
        return lines

    def to_terminal_overline(self) -> str:
        """Grouping role: vinculum as overline above text (Chuquet 1484 style)."""
        top = self.top_text()
        glyph = ROLE_GLYPH[VinculumRole.GROUPING]
        bar = glyph * (len(top) + 2)
        return f"{bar}\n {top}"

    def to_inline(self) -> str:
        """Single-line: 'top ∕ bottom' or single-valued representation."""
        top = self.top_text()
        bottom = self.bottom_text()
        if not bottom:
            if self.role == VinculumRole.RADICAL:
                return f"√({top})"
            elif self.role == VinculumRole.COMPLEMENT:
                return f"{top}ᶜ"
            elif self.role == VinculumRole.NEGATION:
                return "".join(ch + "̶" for ch in top)
            elif self.role == VinculumRole.GROUPING:
                return "".join(ch + "\u0305" for ch in top)
            elif self.role == VinculumRole.CONJUGATE:
                return "".join(ch + "\u0304" for ch in top)
            elif self.role == VinculumRole.MULTIPLICATION:
                return f"|{top}|"
            elif self.role == VinculumRole.REPETITION:
                return "".join(ch + "\u0305" for ch in top)
            return top
        return f"{top} ∕ {bottom}"

    # ── Parsing ──
    @classmethod
    def from_shorthand(cls, raw: str) -> "FracNode":
        """Parse shorthand. v2: supports '[a/b] / [c/d]' for nesting.

        Bracket-delimited groups become nested FracNodes.
        Top-level split on ' / ' (slash with surrounding spaces) or first bare '/'.
        """
        raw = raw.strip()

        # Detect bracket-delimited nesting: [top] / [bottom]
        if raw.startswith("[") and "] / [" in raw:
            close_bracket = raw.index("]")
            top_part = raw[1:close_bracket]
            rest = raw[close_bracket + 1:]
            # rest should be " / [bottom]"
            if rest.startswith(" / [") and rest.endswith("]"):
                bottom_part = rest[4:-1]
                return cls(
                    top=cls.from_shorthand(top_part),
                    bottom=cls.from_shorthand(bottom_part),
                )

        # Mixed: str / [nested] or [nested] / str
        if raw.startswith("[") and "] / " in raw:
            close_bracket = raw.index("]")
            top_part = raw[1:close_bracket]
            bottom_part = raw[close_bracket + 4:]  # after "] / "
            top_node = cls.from_shorthand(top_part)
            bottom_node = cls.from_shorthand(bottom_part) if "/" in bottom_part and not bottom_part.startswith("[") else bottom_part
            return cls(top=top_node, bottom=bottom_node)

        if " / [" in raw and raw.endswith("]"):
            split_idx = raw.index(" / [")
            top_part = raw[:split_idx]
            bottom_part = raw[split_idx + 4:-1]
            return cls(
                top=top_part.strip(),
                bottom=cls.from_shorthand(bottom_part),
            )

        # Standard: first / only
        parts = raw.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid FracType shorthand: {raw!r}")
        return cls(top=parts[0].strip(), bottom=parts[1].strip())


@dataclass
class FracTable:
    """Multi-fraction container — renders compact aligned vinculum blocks."""
    nodes: list[FracNode] = field(default_factory=list)

    @classmethod
    def from_block(cls, text: str) -> "FracTable":
        nodes = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            nodes.append(FracNode.from_shorthand(line))
        if not nodes:
            raise ValueError(f"No valid FracType shorthands in block: {text!r}")
        return cls(nodes=nodes)

    @property
    def _col_width(self) -> int:
        return max((max(len(n.top_text()), len(n.bottom_text())) for n in self.nodes), default=0) + 2

    def to_terminal_lines(self) -> list[str]:
        w = self._col_width
        lines = []
        for i, node in enumerate(self.nodes):
            if i > 0:
                lines.append("")
            if node.title:
                lines.append(f"[{node.title}]")
            if not node.bottom:
                node_lines = node.to_terminal_lines()
                for line in node_lines:
                    if line.strip() != f"[{node.title}]":
                        lines.append(line.center(w))
            else:
                glyph = ROLE_GLYPH.get(node.role, "─")
                if node.role in (VinculumRole.COMPLEMENT, VinculumRole.NEGATION):
                    glyph = "─"
                lines.append(node.top_text().center(w))
                lines.append(glyph * w)
                lines.append(node.bottom_text().center(w))
        return lines

    def to_terminal_table(self) -> list[str]:
        w = self._col_width
        tops = [n.top_text().center(w) for n in self.nodes]
        bars = []
        for n in self.nodes:
            g = ROLE_GLYPH.get(n.role, "─")
            if n.role in (VinculumRole.COMPLEMENT, VinculumRole.NEGATION):
                g = "─"
            if not n.bottom:
                if n.role == VinculumRole.DIVISION:
                    bars.append("─" * w)
                else:
                    bars.append(" " * w)
            else:
                bars.append(g * w)
        bottoms = [n.bottom_text().center(w) for n in self.nodes]
        sep = " │ "
        return [
            sep.join(tops),
            sep.join(bars),
            sep.join(bottoms),
        ]

    def to_react(self) -> str:
        inner = "\n  ".join(n.to_react() for n in self.nodes)
        return f"<>\n  {inner}\n</>"

    def to_latex(self) -> str:
        return "\\\\[2pt]\n".join(n.to_latex() for n in self.nodes)


# ── Compress-to-Budget Engine ────────────────────────────────────────

def compress_to_budget(pairs: list[tuple[str, str]], max_chars: int,
                       role: VinculumRole = VinculumRole.DIVISION) -> str:
    """Compress a list of (top, bottom) pairs into the tightest vinculum
    representation that fits within max_chars.

    Strategy:
      1. Try side-by-side FracTable (most compact).
      2. Try stacked FracTable.
      3. Try nested vinculum: fold pairs into compound fractions.
      4. Return the tightest fit, or the tightest possible if over budget.
    """
    nodes = [FracNode(top=t, bottom=b, role=role) for t, b in pairs]
    candidates = []

    # ── Candidate 1: side-by-side table (fastest visual parse) ──
    table = FracTable(nodes=nodes)
    table_lines = table.to_terminal_table()
    table_str = "\n".join(table_lines)
    candidates.append(("side-by-side", table_str, len(table_str)))

    # ── Candidate 2: stacked vinculum blocks ──
    stacked_lines = table.to_terminal_lines()
    stacked_str = "\n".join(stacked_lines)
    candidates.append(("stacked", stacked_str, len(stacked_str)))

    # ── Candidate 3: inline slash-compressed sequence ──
    inline_parts = [n.to_inline() for n in nodes]
    inline_str = "  ".join(inline_parts)
    candidates.append(("inline", inline_str, len(inline_str)))

    # ── Candidate 4: nested compression (pairwise folding) ──
    nested = _fold_pairs(nodes, role)
    nested_lines = nested.to_terminal_lines()
    nested_str = "\n".join(nested_lines)
    candidates.append(("nested", nested_str, len(nested_str)))

    # ── Candidate 5: nested-inline ──
    nested_inline = nested.to_inline()
    candidates.append(("nested-inline", nested_inline, len(nested_inline)))

    # ── Pick best fit ──
    # Prefer under-budget, then shortest
    under = [(name, s, cost) for name, s, cost in candidates if cost <= max_chars]
    if under:
        under.sort(key=lambda x: x[2])
        return under[0][1]
    # All over budget — return shortest
    candidates.sort(key=lambda x: x[2])
    return candidates[0][1]


def _fold_pairs(nodes: list[FracNode], role: VinculumRole) -> FracNode:
    """Recursively fold a list of FracNodes into a single nested vinculum.

    For 4 pairs: ((a/b) / (c/d)) / ((e/f) / (g/h)) — binary tree compression.
    For 3 pairs: (a/b) / ((c/d) / (e/f)).
    For 2 pairs: (a/b) / (c/d).
    """
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return FracNode(top=nodes[0], bottom=nodes[1], role=role)
    # Split in half, recurse
    mid = len(nodes) // 2
    left = _fold_pairs(nodes[:mid], role)
    right = _fold_pairs(nodes[mid:], role)
    return FracNode(top=left, bottom=right, role=role)


def compress_text_to_budget(text: str, max_chars: int,
                            role: VinculumRole = VinculumRole.DIVISION) -> str:
    """Take arbitrary text and compress it into a vinculum representation.

    Splits on sentence boundaries, pairs consecutive sentences as top/bottom,
    then runs compress_to_budget.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) < 2:
        return text[:max_chars]
    # Pair consecutive sentences
    pairs = []
    for i in range(0, len(sentences) - 1, 2):
        pairs.append((sentences[i], sentences[i + 1]))
    # Odd sentence gets solo'd
    if len(sentences) % 2 == 1:
        pairs.append((sentences[-1], ""))
    return compress_to_budget(pairs, max_chars, role)


def compress_dict_to_budget(data: dict[str, str], max_chars: int,
                            role: VinculumRole = VinculumRole.DIVISION) -> str:
    """Compress a key/value dict into vinculum pairs."""
    pairs = [(k, v) for k, v in data.items()]
    return compress_to_budget(pairs, max_chars, role)
