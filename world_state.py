from dataclasses import dataclass

import shapely


@dataclass
class WorldState:

    hider_position: shapely.Point | None
    seeker_position: shapely.Point
    frame: int = 0
