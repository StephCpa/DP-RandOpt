"""Countdown: fresh instance generation, exact solver, and RandOpt-compatible rewards.

The reward functions follow RandOpt's ``utils/reward_score/countdown.py``
(main branch, read 2026-10-05): total reward = answer (0/1) + 0.1 * format
(format in {0, 0.1, 0.5, 0.6, 1.0}), so per-example rewards lie in [0, 1.1].
That range IS the sensitivity of a sum-score; either pass ``reward_max=1.1``
to the mechanisms or score privately with ``answer_reward`` alone (range [0, 1]).
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np

COUNTDOWN_REWARD_MAX = 1.1

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ALLOWED = re.compile(r"^[0-9+\-*/()= ]+$")
_FORMULA = re.compile(r"^[0-9+\-*/() ]+$")


# ---------------------------------------------------------------------------
# Rewards (ported)
# ---------------------------------------------------------------------------

def format_reward(response: str) -> float:
    if re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", response, re.DOTALL):
        return 1.0
    r = 0.0
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        r += 0.1
    if re.search(r"<answer>.*?</answer>", response, re.DOTALL):
        r += 0.5
    return r


def _safe_eval(expr: str) -> Optional[Fraction]:
    """Evaluate +,-,*,/ and parentheses exactly, without eval()."""
    tokens = re.findall(r"\d+|[()+\-*/]", expr)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def atom():
        t = peek()
        if t == "(":
            take()
            v = add()
            if take() != ")":
                raise ValueError
            return v
        if t == "-":
            take()
            return -atom()
        if t is not None and t.isdigit():
            return Fraction(int(take()))
        raise ValueError

    def mul():
        v = atom()
        while peek() in ("*", "/"):
            op = take()
            w = atom()
            v = v * w if op == "*" else v / w
        return v

    def add():
        v = mul()
        while peek() in ("+", "-"):
            op = take()
            w = mul()
            v = v + w if op == "+" else v - w
        return v

    try:
        v = add()
        if pos != len(tokens):
            return None
        return v
    except (ValueError, ZeroDivisionError, IndexError):
        return None


def answer_reward(response: str, numbers: List[int], target: int) -> float:
    """1.0 iff the last <answer> uses every number exactly once and reaches the target."""
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return 0.0
    content = matches[-1]
    if not content or not _ALLOWED.match(content):
        return 0.0
    if "=" in content:
        sides = [s.strip() for s in content.split("=") if s.strip()]
        ok = [s for s in sides if _FORMULA.match(s) and sorted(int(n) for n in re.findall(r"\d+", s)) == sorted(numbers)]
        if len(ok) != 1:
            return 0.0
        content = ok[0]
    if sorted(int(n) for n in re.findall(r"\d+", content)) != sorted(numbers):
        return 0.0
    v = _safe_eval(content)
    return 1.0 if v is not None and abs(float(v) - float(target)) < 1e-5 else 0.0


def randopt_reward(response: str, numbers: List[int], target: int) -> float:
    """RandOpt's training reward: answer + 0.1 * format (prepends <think> like RandOpt)."""
    return answer_reward(response, numbers, target) + 0.1 * format_reward("<think>" + response)


def voting_answer(response: str, numbers: List[int]) -> str:
    """Numeric result of a valid formula (RandOpt votes on values), '' if invalid."""
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return ""
    content = matches[-1].strip()
    if not content or not _ALLOWED.match(content):
        return ""
    if "=" in content:
        sides = [s.strip() for s in content.split("=") if s.strip()]
        ok = [s for s in sides if _FORMULA.match(s) and sorted(int(n) for n in re.findall(r"\d+", s)) == sorted(numbers)]
        if len(ok) != 1:
            return ""
        content = ok[0]
    used = [int(n) for n in re.findall(r"\d+", content)]
    if any(n > 1000 for n in used) or sorted(used) != sorted(numbers):
        return ""
    v = _safe_eval(content)
    if v is None:
        return ""
    return str(v.numerator) if v.denominator == 1 else str(float(v))


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def solve(numbers: List[int], target: int) -> Optional[str]:
    """Exact brute-force solver (all orders, all binary trees); None if unsolvable."""
    items = [(Fraction(n), str(n)) for n in numbers]

    def rec(xs):
        if len(xs) == 1:
            return xs[0][1] if xs[0][0] == target else None
        for i, j in itertools.permutations(range(len(xs)), 2):
            (a, sa), (b, sb) = xs[i], xs[j]
            rest = [xs[t] for t in range(len(xs)) if t not in (i, j)]
            cands = [(a + b, f"({sa} + {sb})"), (a - b, f"({sa} - {sb})"), (a * b, f"({sa} * {sb})")]
            if b != 0:
                cands.append((a / b, f"({sa} / {sb})"))
            for v, s in cands:
                out = rec(rest + [(v, s)])
                if out is not None:
                    return out
        return None

    return rec(items)


def generate_instances(count: int, seed: int, n_numbers=(3, 4), max_number: int = 99,
                       max_target: int = 999) -> List[Dict]:
    """Fresh, solvable Countdown instances (the plan's low-contamination source).

    The target is the value of a random expression over all numbers, so every
    instance is solvable; targets outside [1, max_target] or non-integer are redrawn.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        k = int(rng.choice(n_numbers))
        nums = [int(x) for x in rng.integers(1, max_number + 1, size=k)]
        vals = [Fraction(x) for x in nums]
        order = list(rng.permutation(k))
        acc = vals[order[0]]
        for idx in order[1:]:
            op = rng.integers(4)
            v = vals[idx]
            acc = acc + v if op == 0 else acc - v if op == 1 else acc * v if op == 2 else (acc / v)
        if acc.denominator != 1 or not 1 <= acc <= max_target:
            continue
        out.append({"numbers": nums, "target": int(acc)})
    return out


def to_messages(instance: Dict) -> List[Dict]:
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_TEMPLATE.format(numbers=instance["numbers"], target=instance["target"])}]


def encode_answers(answers: List[List[str]], gold: List[str]) -> np.ndarray:
    """Map answer strings [N][T] to ScoreTable codes (0 gold, -1 empty, >=1 per-item wrong)."""
    N, T = len(answers), len(gold)
    codes = np.full((N, T), -1, dtype=np.int16)
    for t in range(T):
        table = {gold[t]: 0}
        for i in range(N):
            a = answers[i][t]
            if not a:
                continue
            if a not in table:
                table[a] = len(table)
            codes[i, t] = table[a]
    return codes
