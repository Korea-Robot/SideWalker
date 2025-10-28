#!/usr/bin/env python3

import heapq
import math

class AStarPlanner:
    """
    A* 경로 탐색 알고리즘을 수행하는 클래스
    """
    def __init__(self):
        # 8-방향 이웃 (대각선 포함)
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
        # 이웃으로의 이동 비용 (직선 1, 대각선 sqrt(2))
        self.move_costs = [1, 1, 1, 1, 
                           math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
    
    # grid distance
    def _heuristic(self, r, c, goal_r, goal_c):
        """휴리스틱 함수 (Euclidean distance)"""
        return math.sqrt((r - goal_r)**2 + (c - goal_c)**2)

    # valid 
    def _is_valid(self, r, c, cells_y, cells_x):
        """그리드 인덱스가 유효한 범위 내에 있는지 확인"""
        return 0 <= r < cells_y and 0 <= c < cells_x


    # grid r,c to goal rc 
    # f = g + h 
    # g : start poit ~ current point cost  (past path)
    # h : current point ~ goal point pred cost (line)
    # priority queue heapq 
    def plan(self, start_rc, goal_rc, costmap, grid_shape):
        """
        A* 알고리즘으로 start (r, c)에서 goal (r, c)까지의 경로 탐색
        
        :param start_rc: (row, col) 시작 그리드 인덱스
        :param goal_rc: (row, col) 목표 그리드 인덱스
        :param costmap: 팽창된 비용 맵 (numpy array)
        :param grid_shape: (total_rows, total_cols) 맵의 전체 크기
        :return: 그리드 인덱스 (r, c)의 리스트로 구성된 경로, 실패 시 None
        """
        
        cells_y, cells_x = grid_shape
        start_r, start_c = start_rc
        goal_r, goal_c = goal_rc

        open_list = []
        heapq.heappush(open_list, (0, start_rc)) # (f_cost, (r, c))
        
        came_from = {}
        g_cost = {start_rc: 0}
        f_cost = {start_rc: self._heuristic(start_r, start_c, goal_r, goal_c)}

        while open_list:
            _, current_rc = heapq.heappop(open_list)
            current_r, current_c = current_rc

            if current_rc == goal_rc:
                # 경로 재구성
                path = []
                while current_rc in came_from:
                    path.append(current_rc)
                    current_rc = came_from[current_rc]
                path.append(start_rc)
                return path[::-1] # start -> goal 순서로 반환

            for i, (dr, dc) in enumerate(self.neighbors):
                neighbor_r, neighbor_c = current_r + dr, current_c + dc
                neighbor_rc = (neighbor_r, neighbor_c)
                
                # 그리드 범위 체크
                if not self._is_valid(neighbor_r, neighbor_c, cells_y, cells_x):
                    continue
                    
                # 장애물 체크 (costmap 값이 0보다 크면 장애물)
                if costmap[neighbor_r, neighbor_c] > 0:
                    continue
                    
                new_g_cost = g_cost[current_rc] + self.move_costs[i]
                
                if neighbor_rc not in g_cost or new_g_cost < g_cost[neighbor_rc]:
                    g_cost[neighbor_rc] = new_g_cost
                    new_f_cost = new_g_cost + self._heuristic(neighbor_r, neighbor_c, goal_r, goal_c)
                    f_cost[neighbor_rc] = new_f_cost
                    heapq.heappush(open_list, (new_f_cost, neighbor_rc))
                    came_from[neighbor_rc] = current_rc
        
        # 경로를 찾지 못함
        return None
