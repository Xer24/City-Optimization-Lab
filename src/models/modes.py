from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Hashable

Edge = Tuple[Hashable, Hashable]
modes = ("car", "pedestrian", "public_transit")

@dataclass
class MultiModalFlows:
    car: dict[Edge, float]
    pedestrian: dict[Edge, float]
    public_transit: dict[Edge, float]

    def total(self) -> Dict[Edge, float]:
        out: Dict[Edge, float] = {}
        for d in (self.car, self.pedestrian, self.public_transit):
            for e, f in d.items():
                out[e] = out.get(e,0.0) + float(f)
        return out
