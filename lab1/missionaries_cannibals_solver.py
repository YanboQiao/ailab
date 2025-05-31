#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传教士与野人过河——BFS 与 A* 搜索
Author : ChatGPT
Date   : 2025-04-29
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 状态表示与辅助工具
# ──────────────────────────────────────────────────────────────────────────────
State = Tuple[int, int, int]          # (左岸传教士, 左岸野人, 船位置 0=左 1=右)
Path  = List[State]

START_STATE: State = (3, 3, 0)
GOAL_STATE:  State = (0, 0, 1)

@dataclass(frozen=True, slots=True)
class Node:
    state: State
    g: int
    parent: Optional['Node'] = None

    def __lt__(self, other: 'Node') -> bool:     # pragma: no cover
        return False

# ──────────────────────────────────────────────────────────────────────────────
# 合法性判断与后继生成
# ──────────────────────────────────────────────────────────────────────────────
MOVES: Tuple[Tuple[int, int], ...] = ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1))

def is_valid(s: State) -> bool:
    """检查左右两岸是否合法（传教士数量不为负且若有传教士则≥野人）"""
    ml, cl, boat = s
    mr, cr = 3 - ml, 3 - cl
    # 非负
    if min(ml, cl, mr, cr) < 0 or max(ml, cl, mr, cr) > 3:
        return False
    # 左岸合法
    if ml > 0 and ml < cl:
        return False
    # 右岸合法
    if mr > 0 and mr < cr:
        return False
    return True

def successors(s: State) -> List[State]:
    """生成合法后继状态"""
    ml, cl, boat = s
    res: List[State] = []
    direction = -1 if boat == 0 else 1   # 0 左→右；1 右→左
    for dm, dc in MOVES:
        if boat == 0:        # 船在左岸
            nxt = (ml - dm, cl - dc, 1)
        else:                # 船在右岸
            nxt = (ml + dm, cl + dc, 0)
        if is_valid(nxt):
            res.append(nxt)
    return res

# ──────────────────────────────────────────────────────────────────────────────
# 启发式函数
# ──────────────────────────────────────────────────────────────────────────────
def heuristic(s: State) -> int:
    """简单启发：左岸剩余人数 - 若船在左岸则不额外加乘，否则再加 1（鼓励尽早把船带回）"""
    ml, cl, boat = s
    return ml + cl + (1 if boat != 0 else 0)

# ──────────────────────────────────────────────────────────────────────────────
# 搜索算法
# ──────────────────────────────────────────────────────────────────────────────
def bfs(start: State, goal: State = GOAL_STATE) -> Tuple[Path, int, float]:
    t0 = time.time()
    root = Node(start, g=0)
    if start == goal:
        return [start], 0, 0.0

    frontier: deque[Node] = deque([root])
    visited: set[State] = {start}
    parent_map: Dict[State, Optional[State]] = {start: None}
    expanded = 0

    while frontier:
        node = frontier.popleft()
        expanded += 1
        for nxt in successors(node.state):
            if nxt in visited:
                continue
            parent_map[nxt] = node.state
            if nxt == goal:
                t1 = time.time()
                return reconstruct_path(nxt, parent_map), expanded, t1 - t0
            visited.add(nxt)
            frontier.append(Node(nxt, node.g + 1))

    raise RuntimeError("No solution found")

def a_star(start: State, goal: State = GOAL_STATE) -> Tuple[Path, int, float]:
    t0 = time.time()
    root = Node(start, g=0)
    open_heap: List[Tuple[int, Node]] = []
    heapq.heappush(open_heap, (heuristic(start), root))
    g_cost: Dict[State, int] = {start: 0}
    parent_map: Dict[State, Optional[State]] = {start: None}
    expanded = 0

    while open_heap:
        f, node = heapq.heappop(open_heap)
        if node.state == goal:
            t1 = time.time()
            return reconstruct_path(node.state, parent_map), expanded, t1 - t0
        expanded += 1
        for nxt in successors(node.state):
            tentative_g = g_cost[node.state] + 1
            if nxt not in g_cost or tentative_g < g_cost[nxt]:
                g_cost[nxt] = tentative_g
                parent_map[nxt] = node.state
                heapq.heappush(open_heap, (tentative_g + heuristic(nxt), Node(nxt, tentative_g)))

    raise RuntimeError("No solution found")

def reconstruct_path(end: State, parent_map: Dict[State, Optional[State]]) -> Path:
    path: Path = []
    cur: Optional[State] = end
    while cur is not None:
        path.append(cur)
        cur = parent_map[cur]
    path.reverse()
    return path

# ──────────────────────────────────────────────────────────────────────────────
# CLI / 测试示例
# ──────────────────────────────────────────────────────────────────────────────
def _print_state(s: State) -> None:
    ml, cl, boat = s
    mr, cr = 3 - ml, 3 - cl
    side = "左" if boat == 0 else "右"
    print(f"[左岸] M={ml}, C={cl} | [右岸] M={mr}, C={cr} | 船:{side}")

def demo() -> None:
    for algo, fn in (("BFS", bfs), ("A*", a_star)):
        print(f"{'='*20} {algo} {'='*20}")
        path, expanded, elapsed = fn(START_STATE)
        print(f"步数: {len(path) - 1}, 扩展节点: {expanded}, 耗时: {elapsed:.6f}s")
        print("路径展示：")
        for s in path:
            _print_state(s)

if __name__ == "__main__":
    demo()
