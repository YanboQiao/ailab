#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
八数码难题——BFS 与 A* 搜索
Author : qyb
Date   : 2025-04-29
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 数据结构与辅助函数
# ──────────────────────────────────────────────────────────────────────────────
Board      = Tuple[Tuple[int, ...], ...]           # 3×3 棋盘，每格 0-8
Coord      = Tuple[int, int]                       # 行、列坐标
Path       = List[Board]

GOAL_STATE: Board = ((1, 2, 3),
                     (4, 5, 6),
                     (7, 8, 0))

@dataclass(frozen=True, slots=True)
class Node:
    board: Board
    g: int                                       # 已耗费成本 g
    parent: Optional['Node'] = None              # 回溯路径

    # 用于 A* 优先队列比较（heapq 依据元组从左到右比较）
    def __lt__(self, other: 'Node') -> bool:     # pragma: no cover
        return False                             # f 值外置，实际不比较 Node

# ──────────────────────────────────────────────────────────────────────────────
# 棋盘操作
# ──────────────────────────────────────────────────────────────────────────────
DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

def find_zero(board: Board) -> Coord:
    """返回空位 (0) 的坐标"""
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j
    raise ValueError("Invalid board – no zero tile")

def swap(board: Board, a: Coord, b: Coord) -> Board:
    """交换两坐标并返回新棋盘（不修改原布局）"""
    lst = [list(row) for row in board]
    (ai, aj), (bi, bj) = a, b
    lst[ai][aj], lst[bi][bj] = lst[bi][bj], lst[ai][aj]
    return tuple(tuple(row) for row in lst)

def successors(board: Board) -> List[Board]:
    """生成可达后继棋盘"""
    zi, zj = find_zero(board)
    neigh: List[Board] = []
    for di, dj in DIRS:
        ni, nj = zi + di, zj + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            neigh.append(swap(board, (zi, zj), (ni, nj)))
    return neigh

# ──────────────────────────────────────────────────────────────────────────────
# 启发式函数
# ──────────────────────────────────────────────────────────────────────────────
GOAL_POS: Dict[int, Coord] = {GOAL_STATE[i][j]: (i, j)
                              for i in range(3)
                              for j in range(3)}

def manhattan(board: Board) -> int:
    """曼哈顿距离启发：所有数字到目标位置行距+列距之和"""
    dist = 0
    for i in range(3):
        for j in range(3):
            tile = board[i][j]
            if tile != 0:
                gi, gj = GOAL_POS[tile]
                dist += abs(i - gi) + abs(j - gj)
    return dist

# ──────────────────────────────────────────────────────────────────────────────
# 搜索算法
# ──────────────────────────────────────────────────────────────────────────────
def bfs(start: Board, goal: Board = GOAL_STATE) -> Tuple[Path, int, float]:
    """广度优先搜索；返回 (路径, 扩展节点数, 耗时 s)"""
    t0 = time.time()
    root = Node(start, g=0)
    if start == goal:
        return [start], 0, 0.0

    frontier: deque[Node] = deque([root])
    visited: set[Board] = {start}
    parent_map: Dict[Board, Optional[Board]] = {start: None}
    expanded = 0

    while frontier:
        node = frontier.popleft()
        expanded += 1
        for nxt in successors(node.board):
            if nxt in visited:
                continue
            parent_map[nxt] = node.board
            if nxt == goal:
                t1 = time.time()
                return (reconstruct_path(nxt, parent_map), expanded, t1 - t0)
            visited.add(nxt)
            frontier.append(Node(nxt, g=node.g + 1))

    raise RuntimeError("No solution found")

def a_star(start: Board, goal: Board = GOAL_STATE) -> Tuple[Path, int, float]:
    """A* 搜索；返回 (路径, 扩展节点数, 耗时 s)"""
    t0 = time.time()
    root = Node(start, g=0)
    open_heap: List[Tuple[int, Node]] = []
    heapq.heappush(open_heap, (manhattan(start), root))
    g_cost: Dict[Board, int] = {start: 0}
    parent_map: Dict[Board, Optional[Board]] = {start: None}
    expanded = 0

    while open_heap:
        f, node = heapq.heappop(open_heap)
        if node.board == goal:
            t1 = time.time()
            return (reconstruct_path(node.board, parent_map), expanded, t1 - t0)
        expanded += 1
        for nxt in successors(node.board):
            tentative_g = g_cost[node.board] + 1
            if nxt not in g_cost or tentative_g < g_cost[nxt]:
                g_cost[nxt] = tentative_g
                parent_map[nxt] = node.board
                heapq.heappush(open_heap, (tentative_g + manhattan(nxt), Node(nxt, tentative_g)))

    raise RuntimeError("No solution found")

def reconstruct_path(end_board: Board, parent_map: Dict[Board, Optional[Board]]) -> Path:
    """根据 parent_map 回溯完整路径"""
    path: Path = []
    cur: Optional[Board] = end_board
    while cur is not None:
        path.append(cur)
        cur = parent_map[cur]
    path.reverse()
    return path

# ──────────────────────────────────────────────────────────────────────────────
# CLI / 测试示例
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_START: Board = ((1, 2, 3),
                        (4, 0, 6),
                        (7, 5, 8))

def _print_board(b: Board) -> None:
    for row in b:
        print(' '.join(str(x) for x in row))
    print('-' * 7)

def demo() -> None:
    """示例运行：比较 BFS 与 A*"""
    for algo, fn in (("BFS", bfs), ("A*", a_star)):
        print(f"{'='*20} {algo} {'='*20}")
        path, expanded, elapsed = fn(DEFAULT_START)
        print(f"步数: {len(path) - 1}, 扩展节点: {expanded}, 耗时: {elapsed:.4f}s")
        print("路径展示：")
        for b in path:
            _print_board(b)

if __name__ == "__main__":
    demo()