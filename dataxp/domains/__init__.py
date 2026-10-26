"""Domain registry. A variant is a (domain module, class, stochastic flag).

Import is lazy so one broken domain cannot take down the others' generation
jobs. Every domain lives in its own module and touches only dataxp.domain /
dataxp.potential.
"""

from __future__ import annotations

import importlib

# variant name -> (module, class name, stochastic)
VARIANTS = {
    "seir":          ("dataxp.domains.seir", "SEIRDomain", False),
    "seir_stoch":    ("dataxp.domains.seir", "SEIRDomain", True),
    "grid":          ("dataxp.domains.grid", "GridDomain", False),
    "grid_stoch":    ("dataxp.domains.grid", "GridDomain", True),
    "thermal":       ("dataxp.domains.thermal", "ThermalDomain", False),
    "thermal_stoch": ("dataxp.domains.thermal", "ThermalDomain", True),
    "traffic":       ("dataxp.domains.traffic", "TrafficDomain", False),
    "traffic_stoch": ("dataxp.domains.traffic", "TrafficDomain", True),
    "inventory":     ("dataxp.domains.inventory", "InventoryDomain", True),
    "circuit":       ("dataxp.domains.circuit", "CircuitDomain", False),
    "circuit_stoch": ("dataxp.domains.circuit", "CircuitDomain", True),
    "frame3d":       ("dataxp.domains.frame3d", "Frame3DDomain", False),
    "frame3d_stoch": ("dataxp.domains.frame3d", "Frame3DDomain", True),
    # v2: full action sets (topology/shape/categorical/toggle moves as typed
    # params). v1 modules above are untouched so run1 replays exactly.
    "seir2":          ("dataxp.domains.seir2", "SEIR2Domain", False),
    "seir2_stoch":    ("dataxp.domains.seir2", "SEIR2Domain", True),
    "grid2":          ("dataxp.domains.grid2", "Grid2Domain", False),
    "grid2_stoch":    ("dataxp.domains.grid2", "Grid2Domain", True),
    "thermal2":       ("dataxp.domains.thermal2", "Thermal2Domain", False),
    "thermal2_stoch": ("dataxp.domains.thermal2", "Thermal2Domain", True),
    "traffic2":       ("dataxp.domains.traffic2", "Traffic2Domain", False),
    "traffic2_stoch": ("dataxp.domains.traffic2", "Traffic2Domain", True),
    "inventory2":     ("dataxp.domains.inventory2", "Inventory2Domain", True),
    "circuit2":       ("dataxp.domains.circuit2", "Circuit2Domain", False),
    "circuit2_stoch": ("dataxp.domains.circuit2", "Circuit2Domain", True),
    "frame3d2":       ("dataxp.domains.frame3d2", "Frame3D2Domain", False),
    "frame3d2_stoch": ("dataxp.domains.frame3d2", "Frame3D2Domain", True),
    # native-verb domains (docs/corpus_native_domains.md, contract in
    # docs/dataxp_move_contract.md). Pre-registered: the modules land as their
    # implementers finish them, and the lazy import means a missing module
    # fails only its own variant (ModuleNotFoundError in `smoke all`).
    "grid2op":        ("dataxp.domains.grid2op", "Grid2OpDomain", False),
    "photonics":      ("dataxp.domains.photonics", "PhotonicsDomain", False),
    "molopt":         ("dataxp.domains.molopt", "MolOptDomain", False),
    "engibench":      ("dataxp.domains.engibench", "EngiBenchDomain", False),
}


def get_domain(variant: str):
    if variant not in VARIANTS:
        raise KeyError("unknown variant %r; have %s" % (variant, sorted(VARIANTS)))
    module_name, class_name, stochastic = VARIANTS[variant]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(stochastic=stochastic)


def list_variants():
    return sorted(VARIANTS)
