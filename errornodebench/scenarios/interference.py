"""Interference scenario task families.

Design goal: force the consolidator to confront tasks whose SURFACE features
are similar but whose applicability conditions differ. The paper's
interference failure mode is that consolidation strips applicability
conditions and the resulting lesson misfires across family boundaries.

Five families, three tasks each = 15 tasks:

  heat-melt   solid -> liquid via heat (butter, chocolate, cheese)
  heat-boil   liquid agitation / phase transition via heat (water, soup, milk-to-simmer)
  heat-cook   chemical transformation of food via heat (egg, rice, steak)
  cool        temperature reduction via cooling appliance (ice, soda, ganache)
  mix         combine ingredients without phase change (dough, eggs+milk, vinaigrette)

The three heat sub-families share "apply heat to a substance" but their
applicability conditions differ on temperature, technique, and goal — that
is where over-generalization shows up. `cool` and `mix` are non-heat
controls that test cross-family contamination.

`default_sequence()` interleaves families so the Cumulative arm hits a
family, leaves it, and returns to it after intervening tasks — matching the
paper's switch-sequence stressor.
"""

from __future__ import annotations

from errornodebench.models import Task


# -------- heat-melt: solid-to-liquid via gentle heat --------
HEAT_MELT_TASKS: list[Task] = [
    Task(
        task_id="heat-melt-01",
        family="heat-melt",
        goal="Melt the stick of butter into liquid.",
        environment="Kitchen with a stove, saucepan, freezer, mixing bowl, and a cold stick of butter.",
        correct_strategy=(
            "Place the butter in the saucepan, put the saucepan on the stove, "
            "set the burner to low heat, and wait until the butter is fully liquid."
        ),
        applicability_conditions=(
            "Use LOW heat ONLY when the goal is solid-to-liquid melting of a "
            "meltable substance. Do not use high heat (burns), do not use "
            "cooling, mixing, or evaporation strategies."
        ),
    ),
    Task(
        task_id="heat-melt-02",
        family="heat-melt",
        goal="Melt the bar of dark chocolate into a smooth liquid for dipping.",
        environment="Kitchen with a stove, a saucepan, a heat-safe bowl, a pot of water, and a chocolate bar.",
        correct_strategy=(
            "Set up a double boiler: pour water into the saucepan, place the "
            "heat-safe bowl on top, add the chocolate to the bowl, and warm "
            "the water on low heat until the chocolate melts. Direct heat "
            "scorches chocolate."
        ),
        applicability_conditions=(
            "For heat-sensitive substances that scorch, use indirect/low heat "
            "(double boiler). Not for general boiling, cooking, or mixing tasks."
        ),
    ),
    Task(
        task_id="heat-melt-03",
        family="heat-melt",
        goal="Melt the block of cheddar cheese for a cheese sauce.",
        environment="Kitchen with a stove, saucepan, whisk, milk, and a block of cheddar.",
        correct_strategy=(
            "Grate the cheese first to speed melting. Warm a splash of milk in "
            "the saucepan on low heat, then whisk in the grated cheese a "
            "handful at a time until smooth."
        ),
        applicability_conditions=(
            "Cheese melts best with grating + low heat + a liquid medium. Not "
            "for tasks that need high heat, freezing, or dry mixing."
        ),
    ),
]


# -------- heat-boil: liquid agitation / phase transition via high heat --------
HEAT_BOIL_TASKS: list[Task] = [
    Task(
        task_id="heat-boil-01",
        family="heat-boil",
        goal="Bring the pot of cold water to a rolling boil.",
        environment="Kitchen with a stove and a pot of cold water.",
        correct_strategy=(
            "Place the pot on the stove burner and turn the burner to high "
            "heat. Wait until bubbles form continuously across the surface."
        ),
        applicability_conditions=(
            "Use HIGH heat for boiling water-like liquids that tolerate it. "
            "Not for cooling, mixing, or for heat-sensitive substances like "
            "chocolate or milk that scorch."
        ),
    ),
    Task(
        task_id="heat-boil-02",
        family="heat-boil",
        goal="Bring the pot of vegetable soup to a vigorous boil.",
        environment="Kitchen with a stove and a pot of cold vegetable soup.",
        correct_strategy=(
            "Place the pot on the stove burner, turn the burner to high heat, "
            "and stir occasionally to prevent scorching at the bottom until "
            "the soup boils."
        ),
        applicability_conditions=(
            "Use high heat plus occasional stirring for thick liquids that "
            "could scorch. Not for dairy (scalds) or chocolate (seizes)."
        ),
    ),
    Task(
        task_id="heat-boil-03",
        family="heat-boil",
        goal="Bring the pot of milk to a bare simmer without scalding it.",
        environment="Kitchen with a stove, a thermometer, and a pot of cold milk.",
        correct_strategy=(
            "Place the pot on the stove burner, set the burner to MEDIUM-LOW "
            "heat, and watch closely. Stop heating when small bubbles form at "
            "the edge — milk scalds and skins if it reaches a rolling boil."
        ),
        applicability_conditions=(
            "Dairy needs medium-low heat, never high heat. A rolling boil "
            "ruins the milk. Not applicable to water boiling or soup boiling."
        ),
    ),
]


# -------- heat-cook: chemical transformation of food via heat --------
HEAT_COOK_TASKS: list[Task] = [
    Task(
        task_id="heat-cook-01",
        family="heat-cook",
        goal="Fry the raw egg sunny-side-up.",
        environment="Kitchen with a stove, a non-stick pan, butter, and a raw egg.",
        correct_strategy=(
            "Place the pan on the stove on medium heat, melt a small amount "
            "of butter, crack the egg into the pan, and cook 2-3 minutes "
            "until the white is set but the yolk is still runny."
        ),
        applicability_conditions=(
            "Use medium heat with fat for frying eggs. Not for boiling, "
            "melting solids alone, or for tasks requiring liquid medium."
        ),
    ),
    Task(
        task_id="heat-cook-02",
        family="heat-cook",
        goal="Cook the cup of rice until tender.",
        environment="Kitchen with a stove, a pot with a lid, water, and a cup of dry rice.",
        correct_strategy=(
            "Add 2 cups water and the rice to the pot, bring to a boil on "
            "high heat, then immediately reduce to low heat, cover, and "
            "simmer 18 minutes. Do not lift the lid during simmering."
        ),
        applicability_conditions=(
            "Rice needs a high-then-low heat pattern plus a covered pot for "
            "steam absorption. Not for dry-heat cooking or for tasks where "
            "the substance must remain liquid."
        ),
    ),
    Task(
        task_id="heat-cook-03",
        family="heat-cook",
        goal="Sear the raw steak to medium-rare with a crusted exterior.",
        environment="Kitchen with a stove, a heavy cast-iron pan, oil with a high smoke point, and a raw steak.",
        correct_strategy=(
            "Heat the cast-iron pan on HIGH heat until smoking. Add oil, then "
            "place the steak in the pan and sear 2 minutes per side without "
            "moving it. Rest the steak after."
        ),
        applicability_conditions=(
            "Searing requires very high dry heat and a heavy pan. Not for "
            "dairy, chocolate, eggs, or any substance that scorches at high "
            "temperatures."
        ),
    ),
]


# -------- cool: temperature reduction (non-heat control family) --------
COOL_TASKS: list[Task] = [
    Task(
        task_id="cool-01",
        family="cool",
        goal="Freeze the cup of water into a solid block of ice.",
        environment="Kitchen with a freezer, a refrigerator, an ice tray, and a cup of room-temperature water.",
        correct_strategy=(
            "Pour the water into the ice tray and place the tray in the freezer "
            "until fully solidified."
        ),
        applicability_conditions=(
            "Use a freezer for liquid-to-solid freezing. Never apply heat — "
            "heat would prevent or reverse the desired phase change."
        ),
    ),
    Task(
        task_id="cool-02",
        family="cool",
        goal="Chill the warm bottle of soda until it is cold to the touch.",
        environment="Kitchen with a refrigerator, a freezer, and a warm bottle of soda.",
        correct_strategy=(
            "Place the bottle in the refrigerator for at least 30 minutes. "
            "Do not use the freezer for soda — it may rupture the pressurized "
            "bottle."
        ),
        applicability_conditions=(
            "Use refrigerator (not freezer) for chilling pressurized liquids. "
            "Heat is contraindicated."
        ),
    ),
    Task(
        task_id="cool-03",
        family="cool",
        goal="Solidify the warm chocolate ganache into a set chocolate truffle filling.",
        environment="Kitchen with a refrigerator, a freezer, a shallow dish, and a bowl of warm chocolate ganache.",
        correct_strategy=(
            "Pour the ganache into the shallow dish, cover the dish, and "
            "place it in the refrigerator for 2 hours until set."
        ),
        applicability_conditions=(
            "Use refrigerator (not freezer — too brittle) to solidify "
            "ganache. Heat would melt it back to liquid."
        ),
    ),
]


# -------- mix: combine ingredients without phase change --------
MIX_TASKS: list[Task] = [
    Task(
        task_id="mix-01",
        family="mix",
        goal="Combine the flour, water, and salt into a uniform dough.",
        environment="Kitchen with a mixing bowl, wooden spoon, flour, water, salt, a stove, and a freezer.",
        correct_strategy=(
            "Add flour and salt to the bowl, pour in water gradually, and "
            "stir with the spoon until uniform dough forms."
        ),
        applicability_conditions=(
            "Use a container plus stirring tool to combine ingredients into a "
            "uniform mixture. Do NOT apply heat or cooling — neither is needed "
            "and both can ruin the result."
        ),
    ),
    Task(
        task_id="mix-02",
        family="mix",
        goal="Whisk the eggs and milk into a uniform mixture for an omelette base.",
        environment="Kitchen with a mixing bowl, a whisk, raw eggs, and milk.",
        correct_strategy=(
            "Crack the eggs into the bowl, add the milk, and whisk briskly "
            "until the mixture is uniform pale yellow with small bubbles."
        ),
        applicability_conditions=(
            "Use a whisk to combine liquid ingredients into a uniform "
            "emulsion. Do not heat the eggs at this step — the mix step is "
            "separate from any later cooking step."
        ),
    ),
    Task(
        task_id="mix-03",
        family="mix",
        goal="Combine the oil and vinegar into a stable vinaigrette dressing.",
        environment="Kitchen with a small bowl, a whisk, olive oil, and red wine vinegar.",
        correct_strategy=(
            "Pour the vinegar into the bowl, then drizzle the oil slowly "
            "while whisking constantly to form a temporarily-stable emulsion."
        ),
        applicability_conditions=(
            "Use slow oil addition plus constant whisking to emulsify "
            "oil and vinegar. Heat and cooling are not used."
        ),
    ),
]


ALL_FAMILIES: dict[str, list[Task]] = {
    "heat-melt": HEAT_MELT_TASKS,
    "heat-boil": HEAT_BOIL_TASKS,
    "heat-cook": HEAT_COOK_TASKS,
    "cool": COOL_TASKS,
    "mix": MIX_TASKS,
}


def default_sequence() -> list[Task]:
    """Interleaved 15-task switch sequence.

    The order rotates through families so the Cumulative arm has to revisit
    each family after seeing intervening tasks — that revisit is where the
    paper's interference shows up (the lesson from family A picks up
    over-general phrasing after the model has seen B and C).
    """
    return [
        # Round 1: one task from each family
        HEAT_MELT_TASKS[0],
        HEAT_BOIL_TASKS[0],
        HEAT_COOK_TASKS[0],
        COOL_TASKS[0],
        MIX_TASKS[0],
        # Round 2: rotate through again
        HEAT_MELT_TASKS[1],
        HEAT_BOIL_TASKS[1],
        HEAT_COOK_TASKS[1],
        COOL_TASKS[1],
        MIX_TASKS[1],
        # Round 3: one more rotation
        HEAT_MELT_TASKS[2],
        HEAT_BOIL_TASKS[2],
        HEAT_COOK_TASKS[2],
        COOL_TASKS[2],
        MIX_TASKS[2],
    ]


def reversed_sequence() -> list[Task]:
    """Ablation: same 15 tasks in reverse order.

    Used to disambiguate recency-vs-family for the Cumulative-arm collapse
    finding. If Cumulative still collapses to mix-03, the surviving entry
    tracks family identity. If it collapses to heat-melt-01 (the new last
    task), the surviving entry tracks recency.
    """
    return list(reversed(default_sequence()))


def family_blocked_sequence() -> list[Task]:
    """Ablation: all three tasks from each family in a row, families in
    reverse order.

    This is the opposite of the interleaved switch schedule: the
    consolidator never has to revisit a family after seeing intervening
    tasks. We use this to isolate the contribution of family-switching
    (vs. sequence length alone) to the Cumulative collapse.
    """
    out: list[Task] = []
    for tasks in (MIX_TASKS, COOL_TASKS, HEAT_COOK_TASKS, HEAT_BOIL_TASKS, HEAT_MELT_TASKS):
        out.extend(tasks)
    return out


SEQUENCES: dict[str, list[Task]] = {
    "default": default_sequence(),
    "reversed": reversed_sequence(),
    "family-blocked": family_blocked_sequence(),
}


def get_task(task_id: str) -> Task:
    """Look up a single :class:`Task` by its ``task_id`` (raises KeyError if absent)."""
    for tasks in ALL_FAMILIES.values():
        for t in tasks:
            if t.task_id == task_id:
                return t
    raise KeyError(task_id)
