"""Answer matching for MATH, where comparing strings is too strict.

``0.5``, ``\\frac12`` and ``\\dfrac{1}{2}`` are one answer written three ways, and a
model that says ``2\\pi`` has not got ``\\pi \\cdot 2`` wrong. A prediction counts here
when it is *mathematically* equal to the reference; normalised string equality is kept
as the first (and, when sympy cannot read the latex, the only) test.
"""

import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from run.utils.utils_build_prompts import _last_boxed

# single letters are free variables; longer runs must be functions/constants sympy has
_KNOWN_NAMES = {
    "sqrt", "pi", "oo", "log", "ln", "exp", "sin", "cos", "tan", "sec", "csc", "cot",
    "arcsin", "arccos", "arctan", "abs", "I", "E",
}

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# units and decorations a model may append to an otherwise correct answer
_UNITS = (
    r"cm|mm|km|ft|feet|foot|in|inches|inch|m|miles|mi|"
    r"degrees|radians|units|square units|cubic units|"
    r"dollars|cents|hours|minutes|seconds|days|ways|people|students"
)


def _normalize(answer) -> str:
    """A MATH answer stripped of everything presentational."""
    if answer is None:
        return ""
    s = str(answer).strip().replace("\n", " ")
    boxed = _last_boxed(s)                               # \boxed{5} is still just 5
    if boxed is not None:
        s = boxed.strip()
    s = re.sub(r"\\(?:text|mbox|textbf|textrm|rm)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\!|\\,|\\;|\\:|\\quad|\\qquad", "", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\^\{?\\circ\}?", "", s)                 # 90^\circ -> 90
    s = s.replace("\\%", "").replace("%", "")
    # only after something to measure: a bare "m" or "x + m" is a variable, not metres
    s = re.sub(rf"(?<=[\d}}\)])\s*\\?(?:{_UNITS})\b\.?\s*$", "", s, flags=re.I)
    s = s.rstrip(".").strip()
    if s.count("=") == 1 and not s.startswith("="):      # "x = 5" -> "5"
        s = s.split("=")[-1].strip()
    s = s.replace("{,}", ",")                            # 14{,}916, latex for 14,916
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)            # 1,000 -> 1000
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)  # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)         # \frac9{19} -> \frac{9}{19}
    s = re.sub(r"(\\frac\{[^{}]*\})(\d)(?!\d)", r"\1{\2}", s)  # \frac{19}9
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    # "3 1/3" is a mixed number; the space is about to go and would leave "31/3"
    s = re.sub(r"(?<![\d./])(\d+)\s+(\d+)/(\d+)(?![\d./])", r"(\1+\2/\3)", s)
    s = s.replace(" ", "")
    if s.startswith("."):
        s = "0" + s
    return s


def _to_sympy(answer: str):
    """A sympy expression for a latex answer, or None if it is not an expression.

    Deliberately narrow: no antlr-backed latex parser is available, so this handles the
    macros MATH answers actually use and gives up on anything else rather than guessing.
    """
    s = answer
    if not s or re.search(r"[<>]|\\in\b|\\cup|\\text|\\begin|[;]", s):
        return None                                      # sets, systems, prose
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = re.sub(r"\\sqrt\[(\d+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # a whole number in front of a fraction is a mixed number: 1\frac{12}{13} is
    # 1 + 12/13, which implicit multiplication would otherwise read as 1 * 12/13
    s = re.sub(r"(?<![\w.])(\d+)\\frac\{([^{}]+)\}\{([^{}]+)\}",
               r"((\1)+((\2)/(\3)))", s)
    for _ in range(8):                                   # \frac nests
        collapsed = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        if collapsed == s:
            break
        s = collapsed
    if "\\" in s:
        return None                                      # a macro we do not model
    s = s.replace("{", "(").replace("}", ")")
    # whatever is left must be numbers, operators and names sympy knows
    if any(w not in _KNOWN_NAMES for w in re.findall(r"[a-zA-Z]{2,}", s)):
        return None
    try:
        return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def math_equal(prediction, ground_truth) -> bool:
    """Whether a prediction means the same number/expression as the reference."""
    pred, truth = _normalize(prediction), _normalize(ground_truth)
    if not truth:
        return False
    if pred == truth:
        return True
    if not pred:
        return False
    if pred.lower() == truth.lower():
        return True

    a, b = _to_sympy(pred), _to_sympy(truth)
    if a is None or b is None:
        return False
    try:
        diff = a - b
        if diff.is_zero:                      # structural, free
            return True
        if not diff.free_symbols:
            # numeric: settle it by evaluating, which is orders of magnitude cheaper
            # than simplify() and is what decides the vast majority of comparisons
            value = complex(sympy.N(diff, 20))
            if abs(value) > 1e-9:
                return False
        return sympy.simplify(diff) == 0      # only the genuinely close/symbolic cases
    except Exception:
        return False


def math_exact_match_score(prediction, ground_truth) -> float:
    """``exact_match_score``'s signature, so it drops into metric_max_over_ground_truths."""
    return float(math_equal(prediction, ground_truth))
