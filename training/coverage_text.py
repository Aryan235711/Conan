"""Explicit worked comparison: does an alibi interval cover a time window?

The interval probe showed that no model tested (0.5B to 7B) can decide
this, and the original SFT traces only stated conclusions ("covering the
whole window, so ruled out") without the comparison.  Every coverage
decision in training data is now written out the same way:

    1. Rewrite times after midnight by adding 24 to the hour, so every time
       on the night sorts correctly (00:35 becomes 24:35).
    2. Check the start: the alibi must start at or before the window opens.
    3. Check the end: the alibi must end at or after the window closes.
    4. It covers the window only if both checks pass.

Times are minutes since noon, as everywhere else in the generators.
"""

from __future__ import annotations


def clock(t: int) -> str:
    """Minutes since noon -> ordinary clock time, e.g. 755 -> '00:35'."""
    total = (t + 12 * 60) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def night(t: int) -> str:
    """Minutes since noon -> night time that sorts correctly, e.g. 755 -> '24:35'."""
    return f"{12 + t // 60:02d}:{t % 60:02d}"


def _compare(x: int, y: int) -> str:
    """Explain which of two night times is earlier, hours first then minutes."""
    hx, mx, hy, my = 12 + x // 60, x % 60, 12 + y // 60, y % 60
    if x == y:
        return f"{night(x)} and {night(y)} are the same time"
    if hx != hy:
        rel = "<" if hx < hy else ">"
        earlier = night(x) if hx < hy else night(y)
        return f"hours {hx} {rel} {hy}, so {earlier} is earlier"
    rel = "<" if mx < my else ">"
    earlier = night(x) if mx < my else night(y)
    return f"hours are both {hx}; minutes {mx:02d} {rel} {my:02d}, so {earlier} is earlier"


def covers(a: int, b: int, lo: int, hi: int) -> bool:
    return a <= lo and b >= hi


def explain_coverage(a: int, b: int, lo: int, hi: int, who: str = "the alibi") -> list[str]:
    """Lines that work out whether the interval [a, b] covers [lo, hi]."""
    lines = []
    moved = [t for t in (a, b, lo, hi) if clock(t) != night(t)]
    if moved:
        seen = []
        for t in moved:
            if t not in seen:
                seen.append(t)
        rewrites = ", ".join(f"{clock(t)} becomes {night(t)}" for t in seen)
        lines.append(f"Times after midnight get 24 added to the hour so they stay in order: {rewrites}.")
    start_ok = a <= lo
    end_ok = b >= hi
    lines.append(f"Start: {who} starts {night(a)}, the window opens {night(lo)}; "
                 f"{_compare(a, lo)}. Starts at or before the window opens: {'yes' if start_ok else 'no'}.")
    lines.append(f"End: it ends {night(b)}, the window closes {night(hi)}; "
                 f"{_compare(b, hi)}. Ends at or after the window closes: {'yes' if end_ok else 'no'}.")
    if start_ok and end_ok:
        lines.append("Both checks pass, so it covers the whole window.")
    else:
        failed = "start" if not start_ok else "end"
        lines.append(f"The {failed} check fails, so it does not cover the whole window.")
    return lines
