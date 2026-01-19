from agent_base import Agent
from world_state import WorldState
from Mesh.nav_mesh import NavMesh

import math
from queue import LifoQueue, Queue
from threading import Lock, Thread
from Mesh.nav_mesh import NavMesh, NavMeshCell
from agent_base import Agent
from world_state import WorldState
import heapq
import shapely
import random
from queue import PriorityQueue
from collections import defaultdict
from copy import copy

# Albert Lu: Hider

class DumbHider(Agent):


    def __init__(self, world_map: NavMesh, max_speed: float):
        Agent.__init__(self, world_map, max_speed)
        self.name = "Albert Hider"
        self.cell_path: list[NavMeshCell] | None = []
        self.path: list[shapely.Point] = []
        self.new_target = None
        self.target = None

    def act(self, state: WorldState) -> tuple[float, float] | None: # TODO: make it so target doesn't continually update by re-adding self.target
        location = state.hider_position
        
        # remove warning
        if location == None:
            return None

        cur_cell = self.map.find_cell(location)

        if cur_cell == None:
            return None
        
        safety_rating_map = {}

        line_to_end = shapely.linestrings([(location.x, location.y), (state.seeker_position.x, state.seeker_position.y)])
        
        if (not self.path) or (self.target == location) or (location.distance(state.seeker_position) < 100) or self.map.polygon.contains(line_to_end):
            safety_rating_map = self.dijkstra_map(state)

            # best_cell = cur_cell
            # max_dist = safety_rating_map.get(cur_cell, 0)

            # for neighbor in cur_cell.neighbors:
            #     dist_from_seeker = safety_rating_map.get(neighbor, 0)
            #     if dist_from_seeker > max_dist:
            #         max_dist = dist_from_seeker
            #         best_cell = neighbor

            best_cell = None
            max_val = -1 

            for cell, value in safety_rating_map.items():
                if value > max_val and len(cell.neighbors) > 1:
                    max_val = value
                    best_cell = cell

            if best_cell is None:
                return None

            self.prev_target = self.target
            self.target = best_cell.polygon.centroid
            self.path = []
            self.cell_path = []

        if (not self.map.polygon.contains(self.target) or self.target == None):
            return None
            
        if self.path != [] and location == self.path[0]:
            self.path.pop(0)

        # If target is in sight, move directly to target

        line_to_end = shapely.linestrings([(location.x, location.y), (self.target.x, self.target.y)])
        if self.map.polygon.contains(line_to_end):
            self.path = [self.target]
            return self.create_vector(state)
        
        # if there is no current path, make one (or emergency retarget)
        if not self.path:
            self.cell_path = self.astar(state, location, self.target, safety_rating_map)
            if not self.cell_path:
                return None

        if not self.path:
            self.funnel_path(self.target, state)
            
        if (self.path):
            return self.create_vector(state)
            
    def escape_options_modifier(self, cell):
        neighbors_count = len(cell.neighbors)
        if neighbors_count == 3:
            return 1.5  
        if neighbors_count == 2:
            return 1.0
        return 0.1     # corners are bad
        
    def dijkstra_map(self, state: WorldState) -> dict[NavMeshCell, float]:
        # breadth first search

        seeker_pos = state.seeker_position
        seeker_cell = self.map.find_cell(seeker_pos)

        # remove error
        if seeker_cell == None:
            return {}
        
        # cell_val stores: {cell: distance}
        cell_val = {seeker_cell: 0.0}
        frontier = [(0.0, random.random(),seeker_cell)] # (distance, cell)

        while frontier:
            current_dist, tiebreak, current_cell = heapq.heappop(frontier)

            if current_cell == None: #remove error
                continue

            if current_dist > cell_val.get(current_cell, float('inf')):
                continue

            for neighbor in current_cell.neighbors:
                # Distance from seeker to this neighbor
                edge_dist = current_cell.distance(neighbor)

                modifier = self.escape_options_modifier(neighbor)
                modified_dist = modifier*edge_dist
                # print(f"modifier:{modifier}")

                total_dist = modified_dist + current_dist

                if total_dist < cell_val.get(neighbor, float('inf')):
                    cell_val[neighbor] = total_dist
                    heapq.heappush(frontier, (total_dist, random.random(), neighbor))

        return cell_val


    # Create next vector from current location towards next point in self.path
    def create_vector(self, state: WorldState) -> tuple[float, float]:
        location = state.hider_position

        # get rid of warning
        if location is None:
            return (0, 0)

        distance = shapely.distance(location, self.path[0])
        if (distance == 0):
            return (0, 0)
        speed = min((self.max_speed*0.99, distance))
        #print(f"distance:{distance}")
        return ((speed/distance) * (self.path[0].x - location.x), (speed/distance) * (self.path[0].y - location.y))

    def signed_area(self, a, b, c):
        val = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) # cross product instead
        return val
    
    def funnel_path(self, target, state: WorldState) -> None:
        # print("funnel path")
        location = state.hider_position

        if self.cell_path is None: # Should never be None but it clears error
            return None


        edges = []
        prev_pt = location
        for i in range(1, len(self.cell_path)):
            prev_cell = self.cell_path[i-1]
            cur_cell = self.cell_path[i]

            shared_edge = prev_cell.neighbors[cur_cell]
            point1, point2 = shapely.get_point(shared_edge, [0, 1])
            midpoint = shared_edge.centroid

            if (self.signed_area(prev_pt, midpoint, point1) > 0):
                left_point = point1
                right_point = point2
            else:
                left_point = point2
                right_point = point1
            
            prev_pt = midpoint

            edges.append([left_point, right_point])
        
        edges.append([target, target]) # this is a bootleg fix but adding the destination as a length-zero edge seems to help

        path = []
        origin_pt = location
        origin_idx = 0

        left_pt = edges[0][0]
        left_idx = 0

        right_pt = edges[0][1]
        right_idx = 0
        i=1

        while (i < len(edges)): 
            next_left_pt = edges[i][0]
            next_right_pt = edges[i][1]
            
            if (self.signed_area(origin_pt, right_pt, next_right_pt) >= 0):
                if (self.signed_area(origin_pt, left_pt, next_right_pt) <= 0):
                    right_pt=next_right_pt
                    right_idx = i
                else: # right point is left of left point, reset
                    if (left_pt != origin_pt):
                        path.append(left_pt)

                    origin_pt = left_pt
                    origin_idx = left_idx

                    i = origin_idx + 1
                    if i < len(edges):
                        left_pt, right_pt = edges[i][0], edges[i][1]
                        left_idx = right_idx = i
                    continue 

            if (self.signed_area(origin_pt, left_pt, next_left_pt) <= 0):
                if (self.signed_area(origin_pt, right_pt, next_left_pt) >= 0):
                    left_pt=next_left_pt
                    left_idx = i
                else: # right point is left of left point, reset
                    if (right_pt != origin_pt):
                        path.append(right_pt)

                    origin_pt = right_pt
                    origin_idx = right_idx

                    i = origin_idx + 1
                    if i < len(edges):
                        left_pt, right_pt = edges[i][0], edges[i][1]
                        left_idx = right_idx = i
                    continue

            i += 1 #advance

        path.append(target)
        self.path = path
        return None

    def astar(self, state: WorldState, start_point: shapely.Point, end_point: shapely.Point, safety_map: dict) -> list[NavMeshCell] | None: 
        if (not self.map.polygon.contains(end_point)):
            return None
        
        # input points, output list of moves 
        start_cell, end_cell = self.map.find_cell(start_point), self.map.find_cell(end_point)
        if start_cell is None or end_cell is None:
            return None
        found_path: bool = False
        frontier = []
        tie_break: int = 0
        heapq.heappush(frontier, (start_cell.distance(end_cell), tie_break, start_cell)) 
        g_scores = defaultdict(lambda: float('inf'))
        g_scores[start_cell] = 0
        backtracking: dict[NavMeshCell, NavMeshCell | None] = {start_cell: None}
        curr: NavMeshCell = start_cell

        while frontier != []:
            curr_f, _, curr = heapq.heappop(frontier)

            if curr == end_cell:
                found_path = True
                break

            if curr_f > g_scores.get(curr, float('inf')) + curr.distance(end_cell):
                continue

            for neighbor in curr.neighbors:
                tie_break += 1

                safety_val = safety_map.get(neighbor, 1000)
                danger_penalty = 100000 / (safety_val + 0.1)

                line_to_end = shapely.linestrings([(neighbor.polygon.centroid.x, neighbor.polygon.centroid.y), (state.seeker_position.x, state.seeker_position.y)])
                if self.map.polygon.contains(line_to_end):
                    danger_penalty += 1000000

                print(danger_penalty)
                new_cost: float = g_scores[curr] + neighbor.distance(curr) + danger_penalty

                if new_cost < g_scores[neighbor]:
                    heapq.heappush(frontier, (new_cost + neighbor.distance(end_cell), tie_break, neighbor)) 
                    backtracking[neighbor] = curr
                    g_scores[neighbor] = new_cost

        if not found_path: # if the while ends and we never found a path then there are no paths
            return None
        
        last: NavMeshCell | None = curr
        moves: list[NavMeshCell] = []
        while last is not None:
            moves.append(last)
            last = backtracking[last]
    
        return moves[::-1]


    @property
    def is_seeker(self) -> bool:
        return False
