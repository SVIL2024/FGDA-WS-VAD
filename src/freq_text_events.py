"""Event-prompt dictionary and the per-frame prompt-response matrix Psi.

**Why this module exists, and why it is not the same thing as
``freq_text_prompts``.**

``freq_text_prompts`` holds *class labels* phrased as short scene templates
("a scene where people are fighting"). The precondition probe measured what that
buys: the resulting direction is a weak linear anomaly cue (within-video AUC
0.676 XD / 0.702 UCF) that barely clears a random direction (0.650 / 0.685) and
on UCF is level with the free feature-magnitude cue. The reason is structural -
one direction per class is a *video-level* summary, so it cannot say *which*
frames are anomalous.

This module instead builds a dictionary of *event descriptions* - sentences that
describe what is happening ("a man is pointing a gun at people and firing"), not
what the category is called. Each frame is then scored against every prompt,
giving a ``(B, T, P)`` response matrix ``Psi`` that retains the temporal axis.
The two published results this follows:

* LAP (arXiv 2403.01169) shows event prompts beat class labels specifically at
  the segment level: *"the learnable prompts of VadCLIP are based on class
  labels, which leads to a video-level anomaly matching. While our concise
  descriptions of basic suspected anomaly events can effectively match the
  segment-level features."* It reports 86.5 AP on XD-Violence vs VadCLIP's 84.51.
* CPL-VAD (arXiv 2602.17077) shows the pseudo labels this kind of matching
  produces still need temporal consistency enforcement, or they fragment
  ("outliers or inconsistencies across time", "fragmented intervals or short
  spurious spikes"). That is the job of ``freq_text_envelope``.

So this file answers "what is anomalous here?" and the envelope module answers
"for how long, coherently?". Neither is the rank-1 feature-space injection the
original plan proposed.

**Wording.** Prompts are written as concrete, visible events with an actor and an
action, because CLIP is trained on image captions and responds to depictable
content. They are deliberately *not* paraphrases of the category name: a prompt
like "a scene of abuse" mostly encodes the word, not the event, and several such
paraphrases collapse onto nearly the same text embedding (``freq_text_prompts``
reports a mean mutual cosine of 0.57 between its class-direction rows, which is
that collapse).
"""

from freq_text_prompts import DATASET_CLASS_NAMES, DATASET_NUM_CLASS

# ---------------------------------------------------------------------------
# XD-Violence (7 classes, index-aligned with freq_text_prompts.XD_CLASS_NAMES)
# ---------------------------------------------------------------------------
_XD_EVENTS = {
    0: [  # normal
        "people are walking along a street as usual",
        "a quiet room with nobody moving",
        "someone is sitting and talking calmly",
        "an empty street with parked cars",
        "people are standing in line and waiting",
        "a person is walking a dog in a park",
        "a crowd is walking through a market",
        "someone is driving a car normally down a road",
    ],
    1: [  # fighting
        "a man is punching another man in the face",
        "two people are punching and kicking each other",
        "a group of people are brawling and throwing punches",
        "a person is being knocked to the ground by a punch",
        "men are wrestling and hitting each other on the floor",
        "someone is swinging a fist at another person",
        "several people are shoving and striking one another",
        "a man is kicking a person who has fallen down",
    ],
    2: [  # shooting
        "a man is pointing a gun and firing it",
        "a person is holding a handgun and shooting",
        "someone is aiming a rifle and pulling the trigger",
        "a man is firing a gun at people nearby",
        "a person is running while shooting a pistol",
        "muzzle flash from a gun being fired in a street",
        "someone is shooting a weapon out of a car window",
        "a man is loading a gun and shooting at a crowd",
    ],
    3: [  # riot
        "a large angry crowd is rioting in the street",
        "people are throwing objects and clashing with police",
        "a mob is running through the streets damaging property",
        "protesters are pushing against a line of police officers",
        "a crowd is shouting and holding up signs angrily",
        "people are setting off flares in a violent crowd",
        "a group of people are rushing at police with shields",
        "a crowd is smashing things during a street protest",
    ],
    4: [  # abuse
        "an adult is hitting a child",
        "a person is grabbing and shaking someone by the arm",
        "a man is pushing a woman against a wall",
        "someone is being dragged and mistreated by another person",
        "a person is slapping another person repeatedly",
        "an adult is yelling at and striking a smaller person",
        "someone is being forced down and held against their will",
        "a man is violently grabbing another person by the neck",
    ],
    5: [  # car accident
        "two cars are crashing into each other on a road",
        "a vehicle collides with another car at an intersection",
        "a car is skidding and hitting a roadside barrier",
        "a truck is crashing into the back of a stopped car",
        "a car is rolling over after a collision",
        "a vehicle is hitting a pedestrian in the road",
        "cars are colliding and debris is flying across the road",
        "a motorcycle is crashing into the side of a car",
    ],
    6: [  # explosion
        "a large fireball and smoke rise from an explosion",
        "a building is blowing apart with flames",
        "a huge blast throws smoke and debris into the air",
        "a car is exploding in a ball of fire",
        "fire and thick black smoke from a sudden blast",
        "a quarry is exploding with a cloud of dust and rock",
        "an explosion sends flames and debris flying outward",
        "a bomb detonates and a wall of fire expands",
    ],
}

# ---------------------------------------------------------------------------
# UCF-Crime (14 classes, index-aligned with freq_text_prompts.UCF_CLASS_NAMES)
# ---------------------------------------------------------------------------
_UCF_EVENTS = {
    0: [  # Normal
        "people are walking along a street as usual",
        "a person is walking normally through a hallway",
        "someone is sitting and talking calmly",
        "people are standing around and chatting",
        "an empty street with parked cars",
        "people are waiting at a bus stop quietly",
        "a person is walking a dog in a park",
        "someone is driving a car normally down a road",
    ],
    1: [  # Abuse
        "an adult is hitting a child",
        "a person is grabbing and shaking someone by the arm",
        "a man is pushing a woman against a wall",
        "someone is being dragged and mistreated by another person",
        "a person is slapping another person repeatedly",
        "an adult is yelling at and striking a smaller person",
        "someone is being forced down and held against their will",
        "a man is violently grabbing another person by the neck",
    ],
    2: [  # Arrest
        "police officers are handcuffing a suspect",
        "a police officer is holding a person down on the ground",
        "a person is being escorted into a police car",
        "several officers are restraining a man on a sidewalk",
        "a suspect is being patted down by an officer",
        "police are leading a handcuffed person away",
        "an officer is pinning a person against a vehicle",
        "a man is being arrested with his hands behind his back",
    ],
    3: [  # Arson
        "a person is pouring liquid and setting a fire",
        "someone is lighting a fire inside a building",
        "flames are spreading across a room and heavy smoke rises",
        "a person is throwing a burning object into a doorway",
        "a fire is burning inside an abandoned building at night",
        "someone is using a lighter to ignite furniture",
        "smoke and flames fill a corridor from a set fire",
        "a person is setting papers alight on the floor",
    ],
    4: [  # Assault
        "a person is punching another person in the street",
        "someone is attacking a person walking by",
        "a man is kicking a person lying on the ground",
        "a person is being shoved and struck by another person",
        "an attacker is slapping and hitting a victim",
        "someone is jumping on another person from behind",
        "a person is throwing punches at someone against a wall",
        "a group is attacking a single person on a sidewalk",
    ],
    5: [  # Burglary
        "a person is climbing through a window into a house",
        "someone is forcing open a door and entering a building",
        "a thief is walking through a room looking for valuables",
        "a person is breaking a window to reach inside",
        "someone is carrying items out of a house at night",
        "a person is crawling through a gap in a fence",
        "someone is prying open a locked door with a tool",
        "a thief is rummaging through drawers inside a house",
    ],
    6: [  # Explosion
        "a large fireball and smoke rise from an explosion",
        "a building is blowing apart with flames",
        "a huge blast throws smoke and debris into the air",
        "fire and thick black smoke from a sudden blast",
        "a quarry is exploding with a cloud of dust and rock",
        "an explosion sends flames and debris flying outward",
        "a bomb detonates and a wall of fire expands",
        "debris and smoke shoot out from a violent blast",
    ],
    7: [  # Fighting
        "a man is punching another man in the face",
        "two people are punching and kicking each other",
        "a group of people are brawling and throwing punches",
        "several people are shoving and striking one another",
        "men are wrestling and hitting each other on the floor",
        "a person is being knocked to the ground by a punch",
        "someone is swinging a fist at another person",
        "a crowd is pushing and trading blows in the street",
    ],
    8: [  # RoadAccidents
        "two cars are crashing into each other on a road",
        "a vehicle collides with another car at an intersection",
        "a car is skidding and hitting a roadside barrier",
        "a car is rolling over after a collision",
        "a vehicle is hitting a pedestrian in the road",
        "a truck is crashing into the back of a stopped car",
        "cars are colliding and debris is flying across the road",
        "a motorcycle is crashing into the side of a car",
    ],
    9: [  # Robbery
        "a person is pointing a gun and demanding money",
        "someone is grabbing a bag and running away",
        "a man is taking cash from a shop at gunpoint",
        "a person is snatching a purse from someone",
        "a thief is holding up a store clerk",
        "someone is forcing a person to hand over valuables",
        "a person is threatening someone and taking their wallet",
        "a robber is running out of a store with stolen goods",
    ],
    10: [  # Shooting
         "a man is pointing a gun and firing it",
         "a person is holding a handgun and shooting",
         "someone is aiming a rifle and pulling the trigger",
         "a man is firing a gun at people nearby",
         "muzzle flash from a gun being fired in a street",
         "a person is running while shooting a pistol",
         "someone is shooting a weapon out of a car window",
         "a man is loading a gun and shooting at a crowd",
     ],
    11: [  # Shoplifting
         "a person is hiding merchandise under their clothes",
         "someone is putting items into a bag in a store aisle",
         "a person is walking out of a shop without paying",
         "someone is slipping goods into a jacket pocket",
         "a person is looking around while taking items off a shelf",
         "someone is concealing a product and leaving the aisle",
         "a person is removing a security tag from an item",
         "someone is filling a backpack with store goods",
     ],
    12: [  # Stealing
         "a person is taking a bicycle that is not theirs",
         "someone is picking up a bag and walking away with it",
         "a person is cutting a lock to take a bike",
         "someone is grabbing luggage and leaving quickly",
         "a person is loading stolen items into a car",
         "someone is taking tools from a truck bed",
         "a person is carrying away an object from a yard",
         "someone is unhooking an item and hiding it",
     ],
    13: [  # Vandalism
         "a person is smashing a window with an object",
         "someone is spray painting graffiti on a wall",
         "a person is kicking over a bin and breaking things",
         "someone is breaking a car window with a tool",
         "a person is tearing down property in the street",
         "someone is throwing rocks at a building",
         "a person is ripping a poster off a wall",
         "someone is damaging a fence with a bat",
     ],
}

DATASET_EVENT_PROMPTS = {'xd': _XD_EVENTS, 'ucf': _UCF_EVENTS}


def _check(dataset: str):
    key = str(dataset).lower()
    if key not in DATASET_EVENT_PROMPTS:
        raise KeyError(f"unknown dataset '{dataset}', expected one of "
                       f"{sorted(DATASET_EVENT_PROMPTS)}")
    return key


def check_alignment():
    """Fail loudly if the event dictionary drifts from the class index tables.

    Both tables are keyed by class index, so a missing or extra class would
    otherwise silently shift every prompt by one position and hand class *i* the
    description of class *i+1* - a bug that degrades results without ever raising.
    """
    problems = []
    for key, table in DATASET_EVENT_PROMPTS.items():
        expected = set(range(DATASET_NUM_CLASS[key]))
        if set(table) != expected:
            problems.append(f'{key}: event classes {sorted(table)} != expected '
                            f'{sorted(expected)}')
        for idx, prompts in table.items():
            if len(prompts) < 8:
                problems.append(f'{key}[{idx}] has only {len(prompts)} prompts')
            if len(set(prompts)) != len(prompts):
                problems.append(f'{key}[{idx}] contains duplicate prompts')
    if problems:
        raise AssertionError('event dictionary misaligned: ' + '; '.join(problems))
    return True


def get_event_prompts(class_idx: int, dataset: str) -> list:
    """Event descriptions for one class index."""
    key = _check(dataset)
    table = DATASET_EVENT_PROMPTS[key]
    if class_idx not in table:
        raise KeyError(f"class index {class_idx} not defined for '{dataset}'")
    return table[class_idx]


def get_normal_events(dataset: str) -> list:
    """The normal-class event descriptions (used as the negative anchor set)."""
    return get_event_prompts(0, dataset)


def get_prompt_owner(dataset: str) -> list:
    """``[class_idx per prompt]`` parallel to :func:`flatten_prompts`.

    Needed because the flattened dictionary - not the class list - is what gets
    encoded and compared: an event prompt belongs to its class only for
    bookkeeping, while the response matrix ``Psi`` is computed over the whole
    dictionary.
    """
    key = _check(dataset)
    owners = []
    for idx in sorted(DATASET_EVENT_PROMPTS[key]):
        owners.extend([idx] * len(DATASET_EVENT_PROMPTS[key][idx]))
    return owners


def flatten_prompts(dataset: str) -> list:
    """All event prompts of a dataset as one flat list, in class order."""
    key = _check(dataset)
    out = []
    for idx in sorted(DATASET_EVENT_PROMPTS[key]):
        out.extend(DATASET_EVENT_PROMPTS[key][idx])
    return out


def prompt_index_map(dataset: str) -> dict:
    """``{class_idx: [slice_start, slice_stop)}`` into :func:`flatten_prompts`."""
    key = _check(dataset)
    out, cursor = {}, 0
    for idx in sorted(DATASET_EVENT_PROMPTS[key]):
        n = len(DATASET_EVENT_PROMPTS[key][idx])
        out[idx] = (cursor, cursor + n)
        cursor += n
    return out


def class_names(dataset: str) -> list:
    return DATASET_CLASS_NAMES[_check(dataset)]


check_alignment()
