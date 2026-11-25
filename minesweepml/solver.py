"""Rule-based Minesweeper solver: the yardstick the network is measured against.

`analyse` works only from what a player sees (the visible board and the total
number of mines) and returns the certain safe cells, the certain mines and
P(mine) for every covered cell:

1. Single-cell rules: a number whose covered neighbours must all be mines, or
   that already has all its mines, settles those neighbours.
2. Subset rules: when one number's unknown cells are a subset of another's,
   the difference holds exactly the difference of their remaining counts.
3. Exact enumeration: the remaining frontier (covered cells next to a
   number) splits into independent components. Each is enumerated by
   backtracking, counting the solutions by their number of mines. The
   components and the interior cells (covered cells next to no number) are
   then combined under the global mine count, weighting every solution by
   the number of ways to place the remaining mines in the interior. That
   gives the exact probability of every cell under a uniform mine layout.

A component that exceeds the node budget is approximated instead, and the
analysis is marked inexact. On standard boards that is rare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from .board import COVERED, neighbours

NODE_BUDGET = 60_000  # backtracking nodes per analysis before falling back to an approximation


@dataclass
class Analysis:
    safe: list[tuple[int, int]] = field(default_factory=list)  # covered cells that cannot hold a mine
    mines: list[tuple[int, int]] = field(default_factory=list)  # covered cells that must hold a mine
    prob: np.ndarray | None = None  # P(mine) per cell; NaN for revealed cells
    exact: bool = True


def analyse(visible: np.ndarray, total_mines: int, budget: int = NODE_BUDGET) -> Analysis:
    rows, cols = visible.shape
    covered = visible == COVERED
    ids = -np.ones((rows, cols), dtype=np.int64)
    cells = [(int(r), int(c)) for r, c in np.argwhere(covered)]
    for i, (r, c) in enumerate(cells):
        ids[r, c] = i
    n = len(cells)

    # Constraints: every revealed number with covered neighbours.
    constraints: list[tuple[list[int], int]] = []
    for r, c in np.argwhere((visible >= 0) & (visible <= 8)):
        nb = [int(ids[nr, nc]) for nr, nc in neighbours(r, c, rows, cols) if covered[nr, nc]]
        if nb:
            constraints.append((nb, int(visible[r, c])))

    known: dict[int, int] = {}  # cell id -> 0 (safe) or 1 (mine)
    _apply_rules(constraints, known, total_mines, n)

    # Reduced constraints over the still unknown cells.
    reduced = []
    for nb, value in constraints:
        unknown = [i for i in nb if i not in known]
        if unknown:
            need = value - sum(known.get(i, 0) for i in nb)
            reduced.append((unknown, need))
    frontier = sorted({i for unknown, _ in reduced for i in unknown})
    frontier_set = set(frontier)
    interior = [i for i in range(n) if i not in known and i not in frontier_set]
    remaining = total_mines - sum(known.values())

    prob = np.full(n, np.nan)
    for i, v in known.items():
        prob[i] = float(v)
    exact = True
    certain_safe = {i for i, v in known.items() if v == 0}
    certain_mine = {i for i, v in known.items() if v == 1}

    components = _components(frontier, reduced)
    solved = []  # (cells, counts {k: ways}, cellsums {k: [ways with a mine per cell]})
    approx = []  # cells of components over the budget
    nodes_left = budget
    for comp_cells, comp_constraints in components:
        result, used = _enumerate(comp_cells, comp_constraints, remaining, nodes_left)
        nodes_left -= used
        if result is None:
            approx.append((comp_cells, comp_constraints))
            exact = False
        else:
            solved.append((comp_cells, *result))

    if exact:
        _combine_exact(solved, interior, remaining, prob, certain_safe, certain_mine)
    else:
        _combine_approx(solved, approx, interior, remaining, prob)

    out_prob = np.full((rows, cols), np.nan)
    for i, (r, c) in enumerate(cells):
        out_prob[r, c] = prob[i]
    return Analysis(
        safe=sorted(cells[i] for i in certain_safe),
        mines=sorted(cells[i] for i in certain_mine),
        prob=out_prob,
        exact=exact,
    )


def _apply_rules(constraints, known: dict[int, int], total_mines: int, n: int) -> None:
    """Single-cell and subset rules until nothing changes, plus the global mine count."""
    changed = True
    while changed:
        changed = False
        reduced = []
        for nb, value in constraints:
            unknown = [i for i in nb if i not in known]
            if not unknown:
                continue
            need = value - sum(known.get(i, 0) for i in nb)
            if need == 0 or need == len(unknown):
                for i in unknown:
                    known[i] = 0 if need == 0 else 1
                changed = True
            else:
                reduced.append((frozenset(unknown), need))
        if changed:
            continue
        # Subset rule over pairs of constraints that share a cell.
        by_cell: dict[int, list[int]] = {}
        for k, (unknown, _) in enumerate(reduced):
            for i in unknown:
                by_cell.setdefault(i, []).append(k)
        seen = set()
        for ks in by_cell.values():
            for a in ks:
                for b in ks:
                    if a == b or (a, b) in seen:
                        continue
                    seen.add((a, b))
                    ua, na = reduced[a]
                    ub, nb_ = reduced[b]
                    if ua < ub:
                        diff, need = ub - ua, nb_ - na
                        if need == 0 or need == len(diff):
                            for i in diff:
                                if i not in known:
                                    known[i] = 0 if need == 0 else 1
                                    changed = True
        if changed:
            continue
        unknown_all = [i for i in range(n) if i not in known]
        left = total_mines - sum(known.values())
        if unknown_all and (left == 0 or left == len(unknown_all)):
            for i in unknown_all:
                known[i] = 0 if left == 0 else 1
            changed = True


def _components(frontier: list[int], reduced) -> list[tuple[list[int], list[tuple[list[int], int]]]]:
    """Split the frontier into groups of cells linked by shared constraints."""
    parent = {i: i for i in frontier}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for unknown, _ in reduced:
        root = find(unknown[0])
        for i in unknown[1:]:
            other = find(i)
            if other != root:
                parent[other] = root
    groups: dict[int, list[int]] = {}
    for i in frontier:
        groups.setdefault(find(i), []).append(i)
    cons: dict[int, list] = {}
    for unknown, need in reduced:
        cons.setdefault(find(unknown[0]), []).append((unknown, need))
    return [(cells, cons[root]) for root, cells in groups.items()]


def _enumerate(cells: list[int], constraints, max_mines: int, budget: int):
    """Count the solutions of one component by number of mines.

    Returns ((counts, cellsums), nodes used), or (None, nodes used) when the
    budget runs out.
    """
    # Order cells so that each one shares constraints with those before it: constraints close early.
    by_cell: dict[int, list[int]] = {i: [] for i in cells}
    for k, (unknown, _) in enumerate(constraints):
        for i in unknown:
            by_cell[i].append(k)
    order, placed = [], set()
    for start in cells:
        if start in placed:
            continue
        stack = [start]
        placed.add(start)
        while stack:
            i = stack.pop(0)
            order.append(i)
            for k in by_cell[i]:
                for j in constraints[k][0]:
                    if j not in placed:
                        placed.add(j)
                        stack.append(j)
    m = len(order)
    cell_cons = [by_cell[i] for i in order]
    need = [c[1] for c in constraints]
    free = [len(c[0]) for c in constraints]
    assign = [0] * m
    counts: dict[int, int] = {}
    sums: dict[int, list[int]] = {}
    nodes = 0

    # Iterative backtracking: value -1 means "not tried yet", then 0, then 1.
    pos, mines = 0, 0
    state = [-1] * m
    while pos >= 0:
        if pos == m:
            counts[mines] = counts.get(mines, 0) + 1
            row = sums.get(mines)
            if row is None:
                row = sums[mines] = [0] * m
            for j in range(m):
                if assign[j]:
                    row[j] += 1
            pos -= 1
            continue
        # undo the previous value of this cell, if any
        v = state[pos]
        if v >= 0:
            for k in cell_cons[pos]:
                need[k] += v
                free[k] += 1
            mines -= v
        v += 1
        if v > 1:
            state[pos] = -1
            pos -= 1
            continue
        state[pos] = v
        nodes += 1
        if nodes > budget:
            return None, nodes
        ok = mines + v <= max_mines
        for k in cell_cons[pos]:
            need[k] -= v
            free[k] -= 1
            if need[k] < 0 or need[k] > free[k]:
                ok = False
        mines += v
        assign[pos] = v
        if ok:
            pos += 1
        # if not ok, stay on this cell: the next loop undoes v and tries the next value
    # map back from the search order to the component's cell order
    index = {cell: j for j, cell in enumerate(order)}
    sums = {k: [row[index[c]] for c in cells] for k, row in sums.items()}
    return (counts, sums), nodes


def _convolve(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] = out.get(ka + kb, 0) + va * vb
    return out


def _combine_exact(solved, interior, remaining, prob, certain_safe, certain_mine) -> None:
    n_int = len(interior)

    def ways_interior(t: int) -> int:
        return comb(n_int, t) if 0 <= t <= n_int else 0

    dists = [counts for _, counts, _ in solved]
    total_dist = {0: 1}
    for d in dists:
        total_dist = _convolve(total_dist, d)
    total = sum(w * ways_interior(remaining - s) for s, w in total_dist.items())
    if total == 0:  # inconsistent board (cannot happen on a real game); leave uniform guesses
        for i in interior:
            prob[i] = remaining / max(1, n_int)
        for cells, _, _ in solved:
            for i in cells:
                prob[i] = 0.5
        return
    for j, (cells, counts, sums) in enumerate(solved):
        others = {0: 1}
        for jj, d in enumerate(dists):
            if jj != j:
                others = _convolve(others, d)
        # weight of "this component has k mines" summed over the rest
        weight = {k: sum(w * ways_interior(remaining - k - s) for s, w in others.items()) for k in counts}
        for x, cell in enumerate(cells):
            num = sum(sums[k][x] * weight[k] for k in counts)
            prob[cell] = num / total
            if num == 0:
                certain_safe.add(cell)
            elif num == total:
                certain_mine.add(cell)
    if n_int:
        num = sum(w * ways_interior(remaining - s) * (remaining - s) for s, w in total_dist.items())
        p = num / (total * n_int)
        for i in interior:
            prob[i] = p
        if num == 0:
            certain_safe.update(interior)
        elif num == total * n_int:
            certain_mine.update(interior)


def _combine_approx(solved, approx, interior, remaining, prob) -> None:
    """Fallback when a component was too large: local estimates, no global coupling."""
    expected = 0.0
    for cells, counts, sums in solved:
        total = sum(counts.values())
        for x, cell in enumerate(cells):
            p = sum(sums[k][x] for k in counts) / total
            prob[cell] = p
            expected += p
    for cells, constraints in approx:
        local: dict[int, list[float]] = {}
        for unknown, need in constraints:
            for i in unknown:
                local.setdefault(i, []).append(need / len(unknown))
        for i in cells:
            p = max(local[i])
            prob[i] = p
            expected += p
    if interior:
        p = min(1.0, max(0.0, (remaining - expected) / len(interior)))
        for i in interior:
            prob[i] = p
