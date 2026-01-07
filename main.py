from copy import copy
from pathlib import Path
import pickle
import sys
import time
import pygame
import pygame.locals
import shapely

import numpy as np

path_to_src = Path(__file__).parent.parent
sys.path.insert(0, str(path_to_src))

from Mesh.nav_mesh import NavMesh
from agents.seeker_agent import DumbSeeker as Seeker
from agent_base import Agent
from agents.hider_agent import DumbHider as Hider
from world_state import WorldState


def main():
    SEEKER_SPEED = 10
    HIDER_SPEED = SEEKER_SPEED * 0.9
    GAME_LENGTH = 240
    HIDER_HEAD_START = 15
    TAG_DISTANCE = 10

    fps = 30
    fps_clock = pygame.time.Clock()

    parent_dir = Path.cwd()
    with open(parent_dir / "maps/maze_5.pkl", "rb") as f:
        mesh: NavMesh = pickle.load(f)
    collider = mesh.polygon.buffer(3)

    min_x, min_y, max_x, max_y = mesh.polygon.bounds
    width, height = max_x - min_x, max_y - min_y

    seeker_position = mesh.random_position()
    seeker: Agent = Seeker(copy(mesh), SEEKER_SPEED)

    while True:
        hider_position = mesh.random_position()
        if not mesh.has_line_of_sight(seeker_position, hider_position):
            break
    hider: Agent = Hider(copy(mesh), HIDER_SPEED)

    seeker._state = WorldState(None, seeker_position)
    hider._state = WorldState(hider_position, seeker_position)

    hider.start()

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    bg = mesh.render(screen, False)

    start_time = time.monotonic()
    frame = 0

    while (clock := time.monotonic() - start_time) < GAME_LENGTH:
        if not seeker.is_alive() and clock >= HIDER_HEAD_START:
            seeker.start()
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(bg, (0, 0))

        # Hider Move
        if hider._next_move:
            with hider._lock:
                dx, dy = hider._next_move
                hider._next_move = None
            d = np.linalg.norm([dx, dy])
            hider_next_position = shapely.affinity.translate(hider_position, dx, dy)
            line = shapely.LineString([hider_position, hider_next_position])
            if d <= HIDER_SPEED and collider.contains(line):
                hider_position = hider_next_position
        with hider._lock:
            hider._state.hider_position = hider_position
            hider._state.seeker_position = seeker_position
            hider._state.frame = frame
        pygame.draw.circle(screen, "#0000ff", hider_position.coords[0], 7)

        # Seeker Move
        if seeker._next_move:
            with seeker._lock:
                dx, dy = seeker._next_move
                seeker._next_move = None
            d = np.linalg.norm([dx, dy])
            seeker_next_position = shapely.affinity.translate(seeker_position, dx, dy)
            line = shapely.LineString([seeker_position, seeker_next_position])
            if d <= SEEKER_SPEED and collider.contains(line):
                seeker_position = seeker_next_position
        with seeker._lock:
            can_see = mesh.has_line_of_sight(seeker_position, hider_position)
            seeker._state.hider_position = hider_position if can_see else None
            seeker._state.seeker_position = seeker_position
            seeker._state.frame = frame
        pygame.draw.circle(screen, "#ff0000", seeker_position.coords[0], 7)

        dist = np.linalg.norm(
            [seeker_position.x - hider_position.x, seeker_position.y - hider_position.y]
        )
        if dist <= TAG_DISTANCE and mesh.has_line_of_sight(
            seeker_position, hider_position
        ):
            print(f"{seeker.name} WINS!")
            pygame.quit()
            sys.exit()
            break

        pygame.display.flip()
        fps_clock.tick(fps)
        frame += 1

    print(f"{hider.name} WINS!")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
