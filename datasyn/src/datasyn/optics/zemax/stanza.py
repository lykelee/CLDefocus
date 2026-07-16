"""
Stanza parser for keyword-form text files.

A "stanza" format consists of lines where:
- Each line has a keyword followed by space-separated parameters
- Indented lines are children of the previous less-indented line
- Quoted strings are treated as single tokens

This format is used by Zemax .zmx files, glass catalog .agf files, and others.

Example:
    KEYWORD param1 param2
    PARENT value
        CHILD1 a b c
        CHILD2 "quoted string"
            GRANDCHILD x
        CHILD3 1.5E-3
    NEXT_PARENT value
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Stanza:
    """
    A single keyword line with its parameters and children.

    Attributes:
        keyword: The first token on the line (identifies the stanza type)
        params: Remaining tokens as raw strings (caller interprets types)
        children: Child stanzas (indented lines below this one)
        line_no: Original line number for error reporting (1-indexed)
    """

    keyword: str
    params: list[str] = field(default_factory=list)
    children: list[Stanza] = field(default_factory=list)
    line_no: int = 0

    def __getitem__(self, key: str) -> list[Stanza]:
        """Get all children with matching keyword."""
        return [c for c in self.children if c.keyword == key]

    def get(self, key: str, default: Stanza | None = None) -> Stanza | None:
        """Get first child with matching keyword, or default if not found."""
        for c in self.children:
            if c.keyword == key:
                return c
        return default

    def has(self, key: str) -> bool:
        """Check if any child has the given keyword."""
        return any(c.keyword == key for c in self.children)

    def get_param(self, index: int, default: str | None = None) -> str | None:
        """Get parameter at index, or default if out of range."""
        if 0 <= index < len(self.params):
            return self.params[index]
        return default

    def get_param_int(self, index: int, default: int | None = None) -> int | None:
        """Get parameter at index as int, or default if invalid."""
        val = self.get_param(index)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def get_param_float(self, index: int, default: float | None = None) -> float | None:
        """Get parameter at index as float, or default if invalid."""
        val = self.get_param(index)
        if val is None:
            return default
        try:
            # Handle special values
            if val.upper() == "INFINITY":
                return float("inf")
            return float(val)
        except ValueError:
            return default

    def __repr__(self) -> str:
        params_str = " ".join(self.params) if self.params else ""
        children_str = f", {len(self.children)} children" if self.children else ""
        return (
            f"Stanza({self.keyword!r} {params_str}{children_str}, line={self.line_no})"
        )


@dataclass
class StanzaDocument:
    """
    Root container for all top-level stanzas in a document.

    Attributes:
        stanzas: List of top-level Stanza objects
        encoding: The encoding used to read the file (if from file)
    """

    stanzas: list[Stanza] = field(default_factory=list)
    encoding: str | None = None

    def __getitem__(self, key: str) -> list[Stanza]:
        """Get all top-level stanzas with matching keyword."""
        return [s for s in self.stanzas if s.keyword == key]

    def get(self, key: str, default: Stanza | None = None) -> Stanza | None:
        """Get first top-level stanza with matching keyword."""
        for s in self.stanzas:
            if s.keyword == key:
                return s
        return default

    def has(self, key: str) -> bool:
        """Check if any top-level stanza has the given keyword."""
        return any(s.keyword == key for s in self.stanzas)

    def __len__(self) -> int:
        return len(self.stanzas)

    def __iter__(self):
        return iter(self.stanzas)


# Regex for tokenizing: matches quoted strings or non-whitespace sequences
_TOKEN_RE = re.compile(r'"([^"]*)"|([\S]+)')


def _tokenize_line(line: str) -> list[str]:
    """
    Split line into tokens, respecting quoted strings.

    - Whitespace separates tokens
    - Quoted strings "..." are single tokens (quotes removed)
    - Empty quotes "" become empty string

    Examples:
        'GLAS N-BK7 0 0' -> ['GLAS', 'N-BK7', '0', '0']
        'CURV 0.0 ""' -> ['CURV', '0.0', '']
        'NAME "John Doe"' -> ['NAME', 'John Doe']
    """
    tokens = []
    for match in _TOKEN_RE.finditer(line):
        # Group 1 is quoted content (without quotes), group 2 is unquoted token
        if match.group(1) is not None:
            tokens.append(match.group(1))
        else:
            tokens.append(match.group(2))
    return tokens


def _get_indent_level(line: str) -> int:
    """
    Get indentation level as count of leading whitespace characters.

    Tabs are treated as single indent units (same as one space for comparison).
    """
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4  # Treat tab as 4 spaces
        else:
            break
    return count


def _is_empty_or_comment(line: str) -> bool:
    """Check if line is empty or a comment."""
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def parse_stanza(text: str) -> StanzaDocument:
    """
    Parse keyword-form text into a StanzaDocument.

    Args:
        text: The text content to parse

    Returns:
        StanzaDocument with hierarchical Stanza objects
    """
    lines = text.splitlines()

    # First pass: collect (line_no, indent, tokens) for non-empty lines
    parsed_lines: list[tuple[int, int, list[str]]] = []

    for i, line in enumerate(lines, start=1):
        if _is_empty_or_comment(line):
            continue

        indent = _get_indent_level(line)
        tokens = _tokenize_line(line)

        if not tokens:
            continue

        parsed_lines.append((i, indent, tokens))

    # Second pass: build tree structure using a stack
    root_stanzas: list[Stanza] = []

    # Stack of (indent_level, Stanza) - tracks parent context
    stack: list[tuple[int, Stanza]] = []

    for line_no, indent, tokens in parsed_lines:
        keyword = tokens[0]
        params = tokens[1:]

        stanza = Stanza(keyword=keyword, params=params, line_no=line_no)

        # Pop stack until we find a parent with smaller indent
        while stack and stack[-1][0] >= indent:
            stack.pop()

        if stack:
            # This stanza is a child of the stack top
            stack[-1][1].children.append(stanza)
        else:
            # This is a top-level stanza
            root_stanzas.append(stanza)

        # Push this stanza onto the stack (it may have children)
        stack.append((indent, stanza))

    return StanzaDocument(stanzas=root_stanzas)


def parse_stanza_file(
    filepath: str | Path,
    encodings: list[str] | None = None,
) -> StanzaDocument:
    """
    Parse a file, trying multiple encodings.

    Args:
        filepath: Path to the file
        encodings: List of encodings to try. Defaults to common encodings
                   including UTF-16 (used by Zemax .zmx files).

    Returns:
        StanzaDocument with hierarchical Stanza objects
    """
    if encodings is None:
        # UTF-16 first since Zemax files commonly use it
        encodings = ["utf-16", "utf-8", "utf-8-sig", "iso-8859-1"]

    filepath = Path(filepath)
    text = None
    used_encoding = None

    for encoding in encodings:
        try:
            text = filepath.read_text(encoding=encoding)
            used_encoding = encoding
            break
        except (UnicodeError, UnicodeDecodeError):
            continue

    if text is None:
        raise UnicodeError(f"Could not decode {filepath} with any of: {encodings}")

    doc = parse_stanza(text)
    doc.encoding = used_encoding
    return doc


# =============================================================================
# Demo / Test
# =============================================================================


def demo():
    """Demo: Parse the example_kwform file."""

    # Find the example file relative to this module
    module_dir = Path(__file__).parent
    example_file = module_dir / "example_kwform"

    if not example_file.exists():
        print(f"Example file not found: {example_file}")
        return

    print(f"Parsing: {example_file}")
    print("=" * 60)

    doc = parse_stanza_file(example_file)
    print(f"Encoding: {doc.encoding}")
    print(f"Top-level stanzas: {len(doc.stanzas)}")
    print()

    for stanza in doc.stanzas:
        _print_stanza(stanza, indent=0)

    print()
    print("=" * 60)
    print("Testing accessor methods:")

    # Test accessor methods
    skills = doc["SKILLS"]
    print(f"Found {len(skills)} SKILLS stanzas")

    for skill in skills:
        print(f"  SKILLS {skill.params}: {len(skill.children)} children")
        for child in skill.children:
            print(f"    {child.keyword}: {child.params}")

    # Test get method
    birth = doc.get("BIRTH")
    if birth:
        year = birth.get_param_int(0)
        month = birth.get_param_int(1)
        day = birth.get_param_int(2)
        print(f"Birth: {year}-{month:02d}-{day:02d}")

    tall = doc.get("TALL")
    if tall:
        unit = tall.get_param(0)
        value = tall.get_param_float(1)
        print(f"Tall: {value} {unit}")


def _print_stanza(stanza: Stanza, indent: int = 0):
    """Helper to print a stanza tree."""
    prefix = "  " * indent
    params_str = " ".join(stanza.params) if stanza.params else "(no params)"
    print(f"{prefix}[{stanza.line_no}] {stanza.keyword}: {params_str}")
    for child in stanza.children:
        _print_stanza(child, indent + 1)


if __name__ == "__main__":
    raise SystemExit(demo())
