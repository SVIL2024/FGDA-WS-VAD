"""Class index tables and CLIP prompts for the Freq-Text Aug direction.

Two gaps in the plan document are closed here.

**"准备文本-类别映射词典" (roadmap step 1).** The document writes
``batch['text_prompts']`` as if every video carried a free-text description.
UCF-Crime and XD-Violence provide only a *category* label, so the category ->
prompt dictionary has to be authored. It is authored here to line up index by
index with VadCLIP's own ``label_map`` (see ``xd_train.py`` /
``ucf_train.py``), so ``prompt_text = get_prompt_text(label_map)`` and the
direction table address the same classes and can never drift apart.

**XD labels are composite.** The ``label`` column of ``list/xd_CLIP_rgb.csv``
holds multi-label strings such as ``B1-B2-0``, ``B2-G-B1`` or a bare ``A``
(VadCLIP's ``get_batch_label`` splits them on ``-``). ``'0'`` is a filler meaning
"no third label", and ``A`` is normal. The *primary* anomaly class - the first
real code in the string - is what the text direction is taken from, because one
video contributes one augmentation direction.

Prompt wording follows the document's template ``"a scene where {text}"``. Each
class gets several paraphrases and their embeddings are averaged, since a single
prompt is dominated by incidental wording rather than by the class concept.
"""

# ---------------------------------------------------------------------------
# normal anchor
# ---------------------------------------------------------------------------
NORMAL_PROMPT = "a normal scene without any incident"

# ---------------------------------------------------------------------------
# XD-Violence: codes and names in VadCLIP's label_map order
#   dict({'A':'normal','B1':'fighting','B2':'shooting','B4':'riot',
#         'B5':'abuse','B6':'car accident','G':'explosion'})
# ---------------------------------------------------------------------------
XD_CODES = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']
XD_CLASS_NAMES = ['normal', 'fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion']
XD_LABEL_TO_IDX = {c: i for i, c in enumerate(XD_CODES)}
XD_LABEL_TO_NAME = dict(zip(XD_CODES, XD_CLASS_NAMES))

# ---------------------------------------------------------------------------
# UCF-Crime: 14 classes in VadCLIP's label_map order
# ---------------------------------------------------------------------------
UCF_CLASS_NAMES = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
                   'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
                   'Shoplifting', 'Stealing', 'Vandalism']
UCF_LABEL_TO_IDX = {n: i for i, n in enumerate(UCF_CLASS_NAMES)}

# ---------------------------------------------------------------------------
# prompts, keyed by class *index* (so the index tables above are the source of
# truth and the two can be cross-checked by length)
# ---------------------------------------------------------------------------
_XD_TEMPLATES = {
    0: [NORMAL_PROMPT, "a ordinary scene of daily life", "a calm scene with nothing unusual"],
    1: ["a scene where people are fighting and punching each other",
        "a scene of a violent brawl between several people",
        "a scene where two people are physical fighting"],
    2: ["a scene where people are shooting guns and firing weapons",
        "a scene of a gunfight with people firing firearms",
        "a scene where someone is shooting a rifle"],
    3: ["a scene where a riot is happening and a violent crowd is rampaging",
        "a scene of a large angry crowd rioting in the street",
        "a scene where a mob is clashing with police"],
    4: ["a scene where a person is being abused and mistreated",
        "a scene of someone violently mistreating another person",
        "a scene where a person is being beaten by an adult"],
    5: ["a scene where a car accident happens and vehicles crash",
        "a scene of two cars colliding on the road",
        "a scene where a vehicle crashes into another car"],
    6: ["a scene where a big explosion happens with fire and smoke",
        "a scene of a huge blast with flames and thick smoke",
        "a scene where a building explodes"],
}

_UCF_TEMPLATES = {
    0: [NORMAL_PROMPT, "a ordinary scene of daily life", "a calm street scene with nothing unusual"],
    1: ["a scene where a person is being abused and mistreated",
        "a scene of someone violently mistreating another person",
        "a scene where a person is being struck by another person"],
    2: ["a scene where police officers are arresting a suspect",
        "a scene of police detaining and handcuffing a person",
        "a scene where officers are apprehending a man on the ground"],
    3: ["a scene where a person is committing arson and setting a fire",
        "a scene of someone deliberately setting a building on fire",
        "a scene where a fire is started intentionally at night"],
    4: ["a scene where a person is being assaulted and attacked",
        "a scene of someone attacking another person with force",
        "a scene where a person is jumped and beaten by others"],
    5: ["a scene where a person is burgling into a building",
        "a scene of someone breaking into a house to steal",
        "a scene where a thief is entering a building illegally"],
    6: ["a scene where a big explosion happens with fire and smoke",
        "a scene of a huge blast with flames and thick smoke",
        "a scene where a quarry explodes"],
    7: ["a scene where people are fighting and punching each other",
        "a scene of a violent brawl between several people",
        "a scene where a group of people are fighting"],
    8: ["a scene where a road accident happens and cars crash",
        "a scene of a traffic collision between vehicles",
        "a scene where a car hits a person on the road"],
    9: ["a scene where a robbery happens and someone is robbed",
        "a scene of someone being robbed at gunpoint",
        "a scene where a thief takes money from a shop"],
    10: ["a scene where people are shooting guns and firing weapons",
         "a scene of a mass shooting with people firing firearms",
         "a scene where someone shoots at a crowd"],
    11: ["a scene where a person is shoplifting in a store",
         "a scene of someone hiding merchandise and walking out of a shop",
         "a scene where a person steals items from a supermarket shelf"],
    12: ["a scene where a person is stealing a bicycle or an object",
         "a scene of someone taking a bike without permission",
         "a scene where a person steals luggage and walks away"],
    13: ["a scene where vandalism happens and property is destroyed",
         "a scene of someone smashing windows and damaging property",
         "a scene where a person kicks over a bin and breaks things"],
}

DATASET_TEMPLATES = {'xd': _XD_TEMPLATES, 'ucf': _UCF_TEMPLATES}
DATASET_NUM_CLASS = {'xd': len(XD_CODES), 'ucf': len(UCF_CLASS_NAMES)}
DATASET_CLASS_NAMES = {'xd': XD_CLASS_NAMES, 'ucf': UCF_CLASS_NAMES}

# ---------------------------------------------------------------------------
# union: XD + UCF joint label space for multi-source training (S1).
# Shared concepts share a code (UCF Fighting->B1, Shooting->B2, Abuse->B5,
# RoadAccidents->B6, Explosion->G); UCF-only classes get H codes. Index order
# = XD order followed by the H classes, so XD rows keep their codes and only
# UCF rows are remapped at list-build time (see build_union_list.py).
# ---------------------------------------------------------------------------
UNION_CLASS_NAMES = XD_CLASS_NAMES + ['arrest', 'arson', 'assault', 'burglary',
                                      'robbery', 'shoplifting', 'stealing',
                                      'vandalism']
UNION_CODES = XD_CODES + ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']
UNION_LABEL_TO_IDX = {c: i for i, c in enumerate(UNION_CODES)}
# UCF class name -> union code ('Normal' handled separately as 'A')
UCF_NAME_TO_UNION_CODE = {
    'Fighting': 'B1', 'Shooting': 'B2', 'Abuse': 'B5', 'RoadAccidents': 'B6',
    'Explosion': 'G', 'Arrest': 'H1', 'Arson': 'H2', 'Assault': 'H3',
    'Burglary': 'H4', 'Robbery': 'H5', 'Shoplifting': 'H6', 'Stealing': 'H7',
    'Vandalism': 'H8',
}
_UNION_TEMPLATES = dict(_XD_TEMPLATES)
_UNION_TEMPLATES.update({
    7: _UCF_TEMPLATES[2],    # arrest
    8: _UCF_TEMPLATES[3],    # arson
    9: _UCF_TEMPLATES[4],    # assault
    10: _UCF_TEMPLATES[5],   # burglary
    11: _UCF_TEMPLATES[9],   # robbery
    12: _UCF_TEMPLATES[11],  # shoplifting
    13: _UCF_TEMPLATES[12],  # stealing
    14: _UCF_TEMPLATES[13],  # vandalism
})
DATASET_TEMPLATES['union'] = _UNION_TEMPLATES
DATASET_NUM_CLASS['union'] = len(UNION_CODES)
DATASET_CLASS_NAMES['union'] = UNION_CLASS_NAMES


def _check(dataset: str):
    key = dataset.lower()
    if key not in DATASET_TEMPLATES:
        raise KeyError(f"unknown dataset '{dataset}', expected one of {sorted(DATASET_TEMPLATES)}")
    return key


def get_class_prompts(class_idx: int, dataset: str) -> list:
    """Prompt paraphrases for one class index."""
    key = _check(dataset)
    if class_idx not in DATASET_TEMPLATES[key]:
        raise KeyError(f"class index {class_idx} not defined for dataset '{dataset}'")
    return DATASET_TEMPLATES[key][class_idx]


def get_normal_prompt(dataset: str) -> str:
    return get_class_prompts(0, dataset)[0]


def get_all_prompts(dataset: str) -> list:
    """``[num_class][n_prompt]`` list of prompts, index-aligned with ``label_map``."""
    key = _check(dataset)
    return [get_class_prompts(i, dataset) for i in range(DATASET_NUM_CLASS[key])]


# ---------------------------------------------------------------------------
# label parsing
# ---------------------------------------------------------------------------
def parse_label(label, dataset: str):
    """Map a raw CSV label to ``(class_idx, is_anomaly, codes)``.

    XD ships composite strings (``'B1-B2-0'``); the *first* recognised code that
    is not ``A``/normal becomes the primary class, which is the class whose text
    direction is used for that video. An unrecognised non-empty label degrades to
    "anomaly with no specific class" (``class_idx = None, is_anomaly = True``) so
    it is never silently treated as normal.
    """
    key = _check(dataset)
    text = str(label).strip()

    if key == 'xd':
        codes = [c for c in text.split('-') if c in XD_LABEL_TO_IDX]
        if not codes:
            return None, text not in ('A', 'Normal', ''), []
        non_normal = [c for c in codes if XD_LABEL_TO_IDX[c] != 0]
        if non_normal:
            return XD_LABEL_TO_IDX[non_normal[0]], True, codes
        return 0, False, codes

    if key == 'union':
        # XD-style composite strings plus the H codes; UCF rows are remapped to
        # codes at list-build time, so only codes arrive here
        codes = [c for c in text.split('-') if c in UNION_LABEL_TO_IDX]
        if not codes:
            return None, text not in ('A', 'Normal', ''), []
        non_normal = [c for c in codes if UNION_LABEL_TO_IDX[c] != 0]
        if non_normal:
            return UNION_LABEL_TO_IDX[non_normal[0]], True, codes
        return 0, False, codes

    idx = UCF_LABEL_TO_IDX.get(text, None)
    if idx is None:
        return None, text != 'Normal', []
    return idx, idx != 0, [text]


def primary_class(label, dataset: str):
    """Convenience wrapper returning only the primary class index (or None)."""
    return parse_label(label, dataset)[0]


def is_anomaly(label, dataset: str) -> bool:
    return parse_label(label, dataset)[1]


def vadclip_label_map(dataset: str) -> dict:
    """The ``label_map`` VadCLIP's trainers expect, so ``prompt_text`` matches."""
    key = _check(dataset)
    if key == 'xd':
        return dict(zip(XD_CODES, XD_CLASS_NAMES))
    if key == 'union':
        return dict(zip(UNION_CODES, UNION_CLASS_NAMES))
    return {n: n for n in UCF_CLASS_NAMES}
