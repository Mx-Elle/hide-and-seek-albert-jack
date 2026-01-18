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
from shapely.ops import unary_union
import time
from collections import defaultdict
import heapq


class DumbSeeker(Agent):

    def __init__(self, world_map: NavMesh, max_speed: float):
        Agent.__init__(self, world_map, max_speed)
        self.name = "Relatively Smart Seeker"
        self.cell_path: list[NavMeshCell] | None = []
        self.path: list[shapely.Point] = []
        self.new_target = None
        self.target = None
        self.dead_ends: set[NavMeshCell] = set()
        for cell in self.map.cells:
            if len(cell.neighbors) == 1:
                self.dead_ends.add(cell)
        self.cells_in_consideration = copy(self.map.cells)
        self.planned_cells = []
        self.cells_been_in = []
        self.thirty_percent_of_cells = int(0.3 * len(self.map.cells))
        self.should_look_at_a_dead_end = 0
        self.dead_end_cell: None | NavMeshCell = None
        self.hider_possible_loc: shapely.Point | shapely.Polygon | shapely.MultiPolygon | None = None # once hider is seen, create geom of where they could be. 

        # self.vertices: set[tuple[float,...]] = set()
        # self.vertices.update(self.map.polygon.exterior.coords)
        # for inter in self.map.polygon.interiors:
        #     self.vertices.update(inter.coords)


    
    def act(self, state: WorldState) -> tuple[float, float] | None: # TODO: make it so target doesn't continually update by re-adding self.target
        location = state.seeker_position
        curr_cell = self.map.find_cell(location)
        self.cells_been_in.append(curr_cell) if curr_cell not in self.cells_been_in else None

        while len(self.planned_cells) >= int(self.thirty_percent_of_cells/3): # 10 percent of cells
            self.cells_in_consideration.add(self.planned_cells.pop(0))

        while len(self.cells_been_in) >= self.thirty_percent_of_cells:
            self.cells_been_in.pop(0)

        if self.hider_possible_loc is not None: # update possible locations
            new_geom = self.map.polygon.intersection(self.hider_possible_loc.buffer(self.max_speed * 0.9))
            if new_geom.geom_type in ['Point', 'Polygon', 'MultiPolygon']:
                self.hider_possible_loc = new_geom # type: ignore
            if curr_cell is not None and self.hider_possible_loc is not None and self.hider_possible_loc.geom_type != 'Point':
                self.hider_possible_loc = self.hider_possible_loc.difference(curr_cell.polygon) # type: ignore
            
        if self.hider_possible_loc is not None and self.hider_possible_loc.geom_type != 'Point':
            if self.hider_possible_loc.geom_type == 'MultiPolygon':
                poly = max(list(self.hider_possible_loc.geoms), key= lambda x: x.area) # type: ignore
            else:
                poly = self.hider_possible_loc
            if poly.geom_type == 'Polygon' and len(poly.exterior.coords) > 2: # type: ignore
                cent = poly.centroid
                self.target = shapely.points(min(poly.exterior.coords, key= lambda x: math.dist((cent.x, cent.y), x))) # type: ignore


        if state.hider_position is not None:
            self.target = state.hider_position
            self.path = [self.target]
            self.hider_possible_loc = copy(state.hider_position)
            return self.create_vector(state)
        
        if self.should_look_at_a_dead_end == 1 and self.dead_end_cell is not None: # currently going to dead end
            can_see = True
            for point in self.dead_end_cell.polygon.exterior.coords:
                if not self.map.has_line_of_sight(location, shapely.points(point)):
                    can_see = False
                    break
            if can_see:
                self.target = None

        if self.target == None or location == self.target:
            if self.should_look_at_a_dead_end >= 5 and len(self.dead_ends) > 5:
                new_target_cell = min(self.dead_ends, key= lambda cell: self.decide_dead_end(location, cell))
                self.dead_end_cell = new_target_cell
                self.should_look_at_a_dead_end = 0
            else:
                new_target_cell = self.find_best_cell(location)
                if new_target_cell is not None:
                    self.cells_in_consideration.remove(new_target_cell)
            
            if new_target_cell is None: # should never be
                return None
            self.planned_cells.append(new_target_cell)
            new_target = new_target_cell.polygon.centroid

            if not self.map.polygon.contains(new_target):
                return None

            self.prev_target = self.target
            self.target = new_target
            self.path = []
            self.cell_path = []
            self.should_look_at_a_dead_end += 1


        if self.path != [] and location == self.path[0]:
            self.path.pop(0)

        # If target is in sight, move directly to target
        line_to_end = shapely.linestrings([(location.x, location.y), (self.target.x, self.target.y)])
        if self.map.polygon.contains(line_to_end):
            self.path = [self.target]
            return self.create_vector(state)

        # if there is no current path, make one
        if not self.path:
            self.cell_path = self.astar(location, self.target)
            if not self.cell_path:
                return None

        if not self.path:
            self.funnel_path(self.target, state)

        if self.path:
            return self.create_vector(state)
        
    def decide_dead_end(self, location: shapely.Point, cell: NavMeshCell):
        distance = math.dist((location.x, location.y), (cell.polygon.centroid.x, cell.polygon.centroid.y))
        weight = 1
        if cell in self.planned_cells:
            weight = 1.5
        return distance * weight

    def find_best_cell(self, location: shapely.Point):
        best = (float('inf'), None)
        for cell in self.cells_in_consideration:
            heuristic = (math.dist((location.x, location.y), (cell.polygon.centroid.x, cell.polygon.centroid.y))) - cell.polygon.area
            if heuristic < best[0]:
                best = (heuristic, cell)
        return best[1]


    # Create next vector from current location towards next point in self.path
    def create_vector(self, state: WorldState) -> tuple[float, float]:
        location = state.seeker_position
        distance = shapely.distance(location, self.path[0])
        if (distance == 0):
            return (0, 0)
        speed = min((self.max_speed*0.9999, distance))
        return ((speed/distance) * (self.path[0].x - location.x), (speed/distance) * (self.path[0].y - location.y))

    def signed_area(self, a, b, c):
        val = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) # cross product instead
        return val
    
    def funnel_path(self, target, state: WorldState) -> None:
        location = state.seeker_position

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

    def astar(self, start_point: shapely.Point, end_point: shapely.Point, add_weight = True) -> list[NavMeshCell] | None: 
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
                weight = 0
                tie_break += 1
                if add_weight:
                    if neighbor in self.cells_been_in:
                        weight = 20
                new_cost: float = g_scores[curr] + neighbor.distance(curr) + weight

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
        return True
