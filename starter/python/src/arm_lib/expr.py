"""arm_lib/expr.py — Parses the small one-variable math expressions
`/arm_sim/integration_step`'s `function` field uses (see
docs/PROJECT2_PENDULARM.md's "The `function` field" section) into a plain
Python callable `f(t: float) -> float`.

Grammar, precedence lowest to highest (exactly as the spec states it):
    expr   := term (('+' | '-') term)*
    term   := power (('*' | '/') power)*
    power  := unary ('^' power)?          # '^' is right-associative
    unary  := '-' unary | atom            # unary minus binds *tighter* than '^'
    atom   := NUMBER | 't' | NAME '(' expr ')' | '(' expr ')'

That last precedence detail is easy to miss: because unary binds tighter
than '^' here, `-2^2` parses as `(-2)^2 == 4`, not the usual math-notation
`-(2^2) == -4`. This module follows the spec's stated order exactly rather
than the conventional one.

Malformed input (empty string, unknown identifier, unbalanced parens,
trailing garbage, adjacent tokens with no operator, a stray character) all
raise `ExprError` with a specific message — never a crash — so the service
handler that calls `parse_function` can turn it straight into a
`result:false` response.

Evaluation-time domain errors (`sqrt` of a negative number, `1/0`, ...) are
deliberately *not* exceptions here: `_safe_*` helpers return `nan`/`inf`
instead of letting Python's stricter-than-C math raise `ValueError`/
`ZeroDivisionError`/`OverflowError`, matching the spec's "let it become
NaN/inf, the same as ordinary floating-point arithmetic."
"""

from __future__ import annotations

import math
import re
from typing import Callable, List, Tuple

FUNCTIONS = {"sin", "cos", "tan", "exp", "sqrt", "ln", "abs"}


class ExprError(ValueError):
    """Raised for anything that keeps `function` from parsing cleanly."""


# -- Tokenizing --------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<number>\d+(\.\d+)?([eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>[+\-*/^()])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)

Token = Tuple[str, str]  # (kind, text) where kind is "number" | "name" | "op"


def _tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise ExprError(f"unrecognized character {text[pos]!r} at position {pos}")
        pos = match.end()
        kind = match.lastgroup
        if kind == "space":
            continue
        tokens.append((kind, match.group()))
    return tokens


# -- Safe math (evaluation-time domain errors become nan/inf, not crashes) ---

def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0:
            return math.nan
        return math.inf if (a > 0) == (math.copysign(1.0, b) > 0) else -math.inf
    return a / b


def _safe_pow(a: float, b: float) -> float:
    try:
        result = a**b
    except (ValueError, OverflowError):
        return math.nan
    if isinstance(result, complex):
        return math.nan
    return result


def _safe_sqrt(x: float) -> float:
    return math.nan if x < 0 else math.sqrt(x)


def _safe_ln(x: float) -> float:
    return math.nan if x <= 0 else math.log(x)


def _safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


_UNARY_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": _safe_exp,
    "sqrt": _safe_sqrt,
    "ln": _safe_ln,
    "abs": abs,
}


# -- Recursive-descent parser, compiling directly to closures over `t` -------

Node = Callable[[float], float]


class _Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> Token:
        token = self._peek()
        if token is None:
            raise ExprError("unexpected end of expression")
        self._pos += 1
        return token

    def parse(self) -> Node:
        if not self._tokens:
            raise ExprError("empty expression")
        node = self._expr()
        if self._peek() is not None:
            kind, text = self._peek()
            raise ExprError(f"unexpected trailing input starting at {text!r}")
        return node

    def _expr(self) -> Node:
        node = self._term()
        while True:
            token = self._peek()
            if token is not None and token[1] in ("+", "-"):
                op = self._advance()[1]
                rhs = self._term()
                node = self._combine_add(node, rhs, op)
            else:
                return node

    def _term(self) -> Node:
        node = self._power()
        while True:
            token = self._peek()
            if token is not None and token[1] in ("*", "/"):
                op = self._advance()[1]
                rhs = self._power()
                node = self._combine_mul(node, rhs, op)
            else:
                return node

    @staticmethod
    def _combine_add(lhs: Node, rhs: Node, op: str) -> Node:
        if op == "+":
            return lambda t: lhs(t) + rhs(t)
        return lambda t: lhs(t) - rhs(t)

    @staticmethod
    def _combine_mul(lhs: Node, rhs: Node, op: str) -> Node:
        if op == "*":
            return lambda t: lhs(t) * rhs(t)
        return lambda t: _safe_div(lhs(t), rhs(t))

    def _power(self) -> Node:
        base = self._unary()
        token = self._peek()
        if token is not None and token[1] == "^":
            self._advance()
            exponent = self._power()  # right-associative: recurse at the same level
            return lambda t: _safe_pow(base(t), exponent(t))
        return base

    def _unary(self) -> Node:
        token = self._peek()
        if token is not None and token[1] == "-":
            self._advance()
            operand = self._unary()
            return lambda t: -operand(t)
        return self._atom()

    def _atom(self) -> Node:
        token = self._peek()
        if token is None:
            raise ExprError("unexpected end of expression")
        kind, text = self._advance()

        if kind == "number":
            value = float(text)
            return lambda t: value

        if kind == "name":
            if text == "t":
                return lambda t: t
            if text in FUNCTIONS:
                self._expect_op("(")
                argument = self._expr()
                self._expect_op(")")
                func = _UNARY_FUNCTIONS[text]
                return lambda t: func(argument(t))
            raise ExprError(f"unknown identifier {text!r}")

        if kind == "op" and text == "(":
            node = self._expr()
            self._expect_op(")")
            return node

        raise ExprError(f"unexpected token {text!r}")

    def _expect_op(self, expected: str) -> None:
        token = self._peek()
        if token is None or token[1] != expected:
            raise ExprError(f"expected {expected!r}")
        self._advance()


def parse_function(text: str) -> Callable[[float], float]:
    """Parse `text` (the `function` field) into `f(t) -> float`.

    Raises `ExprError` (a `ValueError` subclass) with a specific message on
    any malformed input; never raises for a syntactically valid expression,
    no matter what domain errors show up later at evaluation time.
    """
    if not isinstance(text, str):
        raise ExprError("function must be a string")
    tokens = _tokenize(text)
    return _Parser(tokens).parse()
