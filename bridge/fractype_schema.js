/** Canonical FracType schema — Node.js mirror of fractype_schema.py */

const Scale = Object.freeze({
    SMALL: "small",
    NORMAL: "normal",
    LARGE: "large",
});

const VinculumRole = Object.freeze({
    GROUPING: "grouping",          // overline binding expressions (Chuquet 1484)
    DIVISION: "division",          // fraction bar U+2500
    MULTIPLICATION: "multiplication",  // Roman numeral extension U+2550
    REPETITION: "repetition",      // periodic notation U+2504
    RADICAL: "radical",            // square root / radical
    CONJUGATE: "conjugate",        // complex conjugate / mirror
    COMPLEMENT: "complement",      // set complement
    NEGATION: "negation",          // logical negation / strikethrough
});

const ROLE_GLYPH = Object.freeze({
    [VinculumRole.GROUPING]: "‾",
    [VinculumRole.DIVISION]: "─",
    [VinculumRole.MULTIPLICATION]: "═",
    [VinculumRole.REPETITION]: "┄",
    [VinculumRole.RADICAL]: "─",
    [VinculumRole.CONJUGATE]: "‾",
    [VinculumRole.COMPLEMENT]: "ᶜ",
    [VinculumRole.NEGATION]: "̶",
});

const ROLE_GLYPH_ASCII = Object.freeze({
    [VinculumRole.GROUPING]: "_",
    [VinculumRole.DIVISION]: "-",
    [VinculumRole.MULTIPLICATION]: "=",
    [VinculumRole.REPETITION]: "~",
    [VinculumRole.RADICAL]: "r",
    [VinculumRole.CONJUGATE]: "*",
    [VinculumRole.COMPLEMENT]: "c",
    [VinculumRole.NEGATION]: "~",
});

class FracNode {
    /**
     * @param {string|FracNode} top
     * @param {string|FracNode} [bottom=""]
     * @param {string} [title=""]
     * @param {string} [scale="normal"]
     * @param {string} [colorHint="nominal"]
     * @param {string} [role="division"]
     */
    constructor(top, bottom = "", title = "", scale = Scale.NORMAL, colorHint = "nominal", role = VinculumRole.DIVISION) {
        this.top = top;
        this.bottom = bottom;
        this.title = title;
        this.scale = scale;
        this.colorHint = colorHint;
        this.role = role;
    }

    topText() {
        if (this.top instanceof FracNode) {
            return this.top.toInline();
        }
        return String(this.top);
    }

    bottomText() {
        if (!this.bottom) return "";
        if (this.bottom instanceof FracNode) {
            return this.bottom.toInline();
        }
        return String(this.bottom);
    }

    depth() {
        let d = 0;
        if (this.top instanceof FracNode) {
            d = Math.max(d, this.top.depth() + 1);
        }
        if (this.bottom instanceof FracNode) {
            d = Math.max(d, this.bottom.depth() + 1);
        }
        return d;
    }

    costTerminal() {
        const w = Math.max(this.topText().length, this.bottomText().length, this.title.length) + 2;
        let lines = 3;
        if (this.title) lines += 1;
        return w * lines;
    }

    costInline() {
        return this.toInline().length;
    }

    toReact() {
        const top = this.top instanceof FracNode ? this.top.toReact() : `"${this.top}"`;
        if (this.bottom) {
            const bottom = this.bottom instanceof FracNode ? this.bottom.toReact() : `"${this.bottom}"`;
            return `<VinculumTypography top={top} bottom={bottom} scale="${this.scale}" role="${this.role}" />`;
        } else {
            return `<VinculumTypography top={top} scale="${this.scale}" role="${this.role}" />`;
        }
    }

    toLaTeX() {
        const top = this.top instanceof FracNode ? this.top.toLaTeX() : `\\text{${this.top}}`;
        if (this.bottom) {
            const bottom = this.bottom instanceof FracNode ? this.bottom.toLaTeX() : `\\text{${this.bottom}}`;
            if (this.role === VinculumRole.RADICAL) {
                return `\\sqrt[${bottom}]{${top}}`;
            }
            return `\\frac{${top}}{${bottom}}`;
        } else {
            if (this.role === VinculumRole.RADICAL) {
                return `\\sqrt{${top}}`;
            } else if (this.role === VinculumRole.GROUPING || this.role === VinculumRole.CONJUGATE) {
                return `\\overline{${top}}`;
            } else if (this.role === VinculumRole.MULTIPLICATION) {
                return `\\overline{\\overline{${top}}}`;
            } else if (this.role === VinculumRole.REPETITION) {
                return `\\overline{${top}}`;
            } else if (this.role === VinculumRole.COMPLEMENT) {
                return `{${top}}^{c}`;
            } else if (this.role === VinculumRole.NEGATION) {
                return `\\xout{${top}}`;
            }
            return top;
        }
    }

    toTerminalLines(glyph = null) {
        const top = this.topText();
        const bottom = this.bottomText();

        if (bottom) {
            if (glyph === null) {
                glyph = ROLE_GLYPH[this.role] || "─";
            }
            const width = Math.max(top.length, bottom.length, this.title.length) + 2;
            const lines = [];
            if (this.title) lines.push(`[${this.title}]`);
            
            if (this.role === VinculumRole.RADICAL) {
                lines.push(` √ ${top}`.padStart(Math.floor((width + top.length + 3) / 2)).padEnd(width));
                lines.push("─".repeat(width));
                lines.push(bottom.padStart(Math.floor((width + bottom.length) / 2)).padEnd(width));
            } else {
                lines.push(top.padStart(Math.floor((width + top.length) / 2)).padEnd(width));
                lines.push(glyph.repeat(width));
                lines.push(bottom.padStart(Math.floor((width + bottom.length) / 2)).padEnd(width));
            }
            return lines;
        }

        const width = Math.max(top.length, this.title.length) + 2;
        const lines = [];
        if (this.title) lines.push(`[${this.title}]`);

        if (this.role === VinculumRole.DIVISION) {
            lines.push(top.padStart(Math.floor((width + top.length) / 2)).padEnd(width));
            lines.push("─".repeat(width));
            lines.push(" ".repeat(width));
        } else if (this.role === VinculumRole.GROUPING || this.role === VinculumRole.CONJUGATE) {
            const g = ROLE_GLYPH[this.role] || "‾";
            lines.push(g.repeat(top.length));
            lines.push(top);
        } else if (this.role === VinculumRole.MULTIPLICATION) {
            const g = ROLE_GLYPH[this.role] || "═";
            lines.push(g.repeat(top.length));
            lines.push(top);
        } else if (this.role === VinculumRole.REPETITION) {
            const g = ROLE_GLYPH[this.role] || "┄";
            lines.push(g.repeat(top.length));
            lines.push(top);
        } else if (this.role === VinculumRole.RADICAL) {
            lines.push(" ┌" + "─".repeat(top.length));
            lines.push(`√ ${top}`);
        } else if (this.role === VinculumRole.COMPLEMENT) {
            lines.push(`${top}ᶜ`);
        } else if (this.role === VinculumRole.NEGATION) {
            const negated = [...top].map(ch => ch + "̶").join("");
            lines.push(negated);
        } else {
            lines.push(top);
        }

        return lines;
    }

    toInline() {
        const top = this.topText();
        const bottom = this.bottomText();
        if (!bottom) {
            if (this.role === VinculumRole.RADICAL) {
                return `√(${top})`;
            } else if (this.role === VinculumRole.COMPLEMENT) {
                return `${top}ᶜ`;
            } else if (this.role === VinculumRole.NEGATION) {
                return [...top].map(ch => ch + "̶").join("");
            } else if (this.role === VinculumRole.GROUPING) {
                return [...top].map(ch => ch + "\u0305").join("");
            } else if (this.role === VinculumRole.CONJUGATE) {
                return [...top].map(ch => ch + "\u0304").join("");
            } else if (this.role === VinculumRole.MULTIPLICATION) {
                return `|${top}|`;
            } else if (this.role === VinculumRole.REPETITION) {
                return [...top].map(ch => ch + "\u0305").join("");
            }
            return top;
        }
        return `${top} ∕ ${bottom}`;
    }

    static fromShorthand(raw) {
        raw = raw.trim();

        // Detect bracket-delimited nesting: [top] / [bottom]
        if (raw.startsWith("[") && raw.includes("] / [")) {
            const closeBracket = raw.indexOf("]");
            const topPart = raw.substring(1, closeBracket);
            const rest = raw.substring(closeBracket + 1);
            if (rest.startsWith(" / [") && rest.endsWith("]")) {
                const bottomPart = rest.substring(4, rest.length - 1);
                return new FracNode(
                    FracNode.fromShorthand(topPart),
                    FracNode.fromShorthand(bottomPart)
                );
            }
        }

        // Mixed: str / [nested] or [nested] / str
        if (raw.startsWith("[") && raw.includes("] / ")) {
            const closeBracket = raw.indexOf("]");
            const topPart = raw.substring(1, closeBracket);
            const bottomPart = raw.substring(closeBracket + 4);
            const topNode = FracNode.fromShorthand(topPart);
            const bottomNode = (bottomPart.includes("/") && !bottomPart.startsWith("["))
                ? FracNode.fromShorthand(bottomPart)
                : bottomPart;
            return new FracNode(topNode, bottomNode);
        }

        if (raw.includes(" / [") && raw.endsWith("]")) {
            const splitIdx = raw.indexOf(" / [");
            const topPart = raw.substring(0, splitIdx);
            const bottomPart = raw.substring(splitIdx + 4, raw.length - 1);
            return new FracNode(
                topPart.trim(),
                FracNode.fromShorthand(bottomPart)
            );
        }

        const idx = raw.indexOf("/");
        if (idx === -1) throw new Error(`Invalid FracType shorthand: ${raw}`);
        return new FracNode(raw.substring(0, idx).trim(), raw.substring(idx + 1).trim());
    }
}

class FracTable {
    /**
     * @param {FracNode[]} nodes
     */
    constructor(nodes = []) {
        this.nodes = nodes;
    }

    static fromBlock(text) {
        const nodes = [];
        for (const line of text.trim().split(/\r?\n/)) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            nodes.push(FracNode.fromShorthand(trimmed));
        }
        if (nodes.length === 0) throw new Error(`No valid FracType shorthands in block: ${text}`);
        return new FracTable(nodes);
    }

    get _colWidth() {
        let maxW = 0;
        for (const n of this.nodes) {
            maxW = Math.max(maxW, n.topText().length, n.bottomText().length);
        }
        return maxW + 2;
    }

    toTerminalLines() {
        const w = this._colWidth;
        const lines = [];
        for (let i = 0; i < this.nodes.length; i++) {
            const node = this.nodes[i];
            if (i > 0) lines.push("");
            if (node.title) lines.push(`[${node.title}]`);
            if (!node.bottom) {
                const nodeLines = node.toTerminalLines();
                for (const line of nodeLines) {
                    if (line.trim() !== `[${node.title}]`) {
                        lines.push(line.padStart(Math.floor((w + line.length) / 2)).padEnd(w));
                    }
                }
            } else {
                let glyph = ROLE_GLYPH[node.role] || "─";
                if (node.role === VinculumRole.COMPLEMENT || node.role === VinculumRole.NEGATION) {
                    glyph = "─";
                }
                const top = node.topText();
                const bottom = node.bottomText();
                lines.push(top.padStart(Math.floor((w + top.length) / 2)).padEnd(w));
                lines.push(glyph.repeat(w));
                lines.push(bottom.padStart(Math.floor((w + bottom.length) / 2)).padEnd(w));
            }
        }
        return lines;
    }

    toTerminalTable() {
        const w = this._colWidth;
        const tops = this.nodes.map(n => {
            const top = n.topText();
            return top.padStart(Math.floor((w + top.length) / 2)).padEnd(w);
        });
        const bars = this.nodes.map(n => {
            let glyph = ROLE_GLYPH[n.role] || "─";
            if (n.role === VinculumRole.COMPLEMENT || n.role === VinculumRole.NEGATION) {
                glyph = "─";
            }
            if (!n.bottom) {
                if (n.role === VinculumRole.DIVISION) {
                    return "─".repeat(w);
                } else {
                    return " ".repeat(w);
                }
            }
            return glyph.repeat(w);
        });
        const bottoms = this.nodes.map(n => {
            const bottom = n.bottomText();
            return bottom.padStart(Math.floor((w + bottom.length) / 2)).padEnd(w);
        });
        const sep = " │ ";
        return [
            tops.join(sep),
            bars.join(sep),
            bottoms.join(sep),
        ];
    }

    toReact() {
        return `<>\n${this.nodes.map(n => `  ${n.toReact()}`).join("\n")}\n</>`;
    }

    toLaTeX() {
        return this.nodes.map(n => n.toLaTeX()).join("\\\\[2pt]\n");
    }
}

module.exports = { FracNode, FracTable, Scale, VinculumRole, ROLE_GLYPH, ROLE_GLYPH_ASCII };
