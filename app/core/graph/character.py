"""Character detection: skinning, humanoid rig heuristics, classification.

Names are only one signal - structure decides. Bip01/mixamo/Japanese/custom
rigs are matched by bone-graph shape as well as by name patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.types import Classification, ModelAsset, Skeleton

# canonical slot -> regexes that may name it in the wild
BONE_PATTERNS = {
    "root": [r"^root$", r"^bip\d*$", r"^armature$", r"^reference$", r"^hips?_?root"],
    "pelvis": [r"pelvis", r"hips?$", r"bip\d*_?pelvis", r"^koshi", r"waist"],
    "spine": [r"spine", r"bip\d*_?spine", r"^abdomen", r"^back", r"^senaka"],
    "chest": [r"chest", r"spine[23]", r"upperchest", r"^mune", r"thorax"],
    "neck": [r"neck", r"bip\d*_?neck", r"^kubi"],
    "head": [r"head$", r"bip\d*_?head", r"^atama"],
    "shoulder_l": [r"(l|left)[_.]?(clavicle|shoulder)", r"clavicle[_.]?l\b", r"shoulder[_.]?l\b"],
    "shoulder_r": [r"(r|right)[_.]?(clavicle|shoulder)", r"clavicle[_.]?r\b", r"shoulder[_.]?r\b"],
    "arm_l": [r"(l|left)[_.]?(upper)?arm", r"arm[_.]?l\b", r"upperarm[_.]?l"],
    "arm_r": [r"(r|right)[_.]?(upper)?arm", r"arm[_.]?r\b", r"upperarm[_.]?r"],
    "forearm_l": [r"(l|left)[_.]?(fore|lower)?arm", r"forearm[_.]?l\b", r"elbow[_.]?l"],
    "forearm_r": [r"(r|right)[_.]?(fore|lower)?arm", r"forearm[_.]?r\b", r"elbow[_.]?r"],
    "hand_l": [r"(l|left)[_.]?hand", r"hand[_.]?l\b", r"wrist[_.]?l"],
    "hand_r": [r"(r|right)[_.]?hand", r"hand[_.]?r\b", r"wrist[_.]?r"],
    "thigh_l": [r"(l|left)[_.]?(up)?leg", r"(l|left)[_.]?thigh", r"thigh[_.]?l\b", r"upleg[_.]?l"],
    "thigh_r": [r"(r|right)[_.]?(up)?leg", r"(r|right)[_.]?thigh", r"thigh[_.]?r\b", r"upleg[_.]?r"],
    "calf_l": [r"(l|left)[_.]?(low)?leg", r"(l|left)[_.]?calf", r"calf[_.]?l\b", r"shin[_.]?l"],
    "calf_r": [r"(r|right)[_.]?(low)?leg", r"(r|right)[_.]?calf", r"calf[_.]?r\b", r"shin[_.]?r"],
    "foot_l": [r"(l|left)[_.]?foot", r"foot[_.]?l\b", r"ankle[_.]?l"],
    "foot_r": [r"(r|right)[_.]?foot", r"foot[_.]?r\b", r"ankle[_.]?r"],
}
COMPILED = {slot: [re.compile(p, re.I) for p in pats] for slot, pats in BONE_PATTERNS.items()}

CORE_SLOTS = ("pelvis", "spine", "head")
LIMB_PAIRS = (("arm_l", "arm_r"), ("hand_l", "hand_r"), ("thigh_l", "thigh_r"),
              ("foot_l", "foot_r"))

WEAPON_HINTS = re.compile(r"(weapon|gun|rifle|pistol|sword|blade|axe|bow|knife|katana)", re.I)
PROP_HINTS = re.compile(r"(prop|chair|table|barrel|crate|box|lamp|door|item|pickup)", re.I)
ENV_HINTS = re.compile(r"(env|environment|building|terrain|road|wall|floor|tree|rock|level|map)", re.I)
CREATURE_HINTS = re.compile(r"(creature|monster|beast|enemy|zombie|dragon|animal|boss)", re.I)
NPC_HINTS = re.compile(r"(npc|villager|citizen|vendor|merchant|guard)", re.I)
CHAR_HINTS = re.compile(r"(char|character|player|hero|pc_|people|human|body|avatar|skin)", re.I)


@dataclass
class RigAnalysis:
    is_humanoid: bool = False
    score: float = 0.0
    matched: dict = field(default_factory=dict)     # slot -> bone name
    bone_count: int = 0
    depth: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"humanoid": self.is_humanoid, "score": round(self.score, 3),
                "bones": self.bone_count, "depth": self.depth,
                "matched_slots": self.matched, "notes": self.notes}


def analyse_skeleton(skeleton: Skeleton) -> RigAnalysis:
    """Score how humanoid a skeleton looks, by names AND by graph shape."""
    analysis = RigAnalysis(bone_count=len(skeleton.bones))
    if not skeleton.bones:
        return analysis

    for slot, patterns in COMPILED.items():
        for bone in skeleton.bones:
            if any(p.search(bone.name) for p in patterns):
                analysis.matched.setdefault(slot, bone.name)
                break

    name_score = len(analysis.matched) / len(COMPILED)

    # structural signals, independent of naming
    depth = _max_depth(skeleton)
    analysis.depth = depth
    branch_points = sum(1 for i in range(len(skeleton.bones)) if len(skeleton.children(i)) >= 2)
    chains = _long_chains(skeleton, min_len=3)

    structure_score = 0.0
    if 15 <= len(skeleton.bones) <= 400:
        structure_score += 0.25
    elif len(skeleton.bones) > 8:
        structure_score += 0.1
    if depth >= 5:
        structure_score += 0.2
    if branch_points >= 3:
        structure_score += 0.2
    if len(chains) >= 4:                       # two arms + two legs
        structure_score += 0.25
    if _has_symmetric_pairs(skeleton):
        structure_score += 0.1

    analysis.score = min(1.0, 0.6 * name_score + 0.6 * structure_score)
    core_hits = sum(1 for slot in CORE_SLOTS if slot in analysis.matched)
    limb_hits = sum(1 for a, b in LIMB_PAIRS if a in analysis.matched and b in analysis.matched)
    analysis.is_humanoid = (core_hits >= 2 and limb_hits >= 2) or analysis.score >= 0.72
    if not analysis.matched and analysis.score >= 0.6:
        analysis.notes.append("humanoid by structure only (unrecognised bone naming)")
    return analysis


def _max_depth(skeleton: Skeleton) -> int:
    depths = {}

    def depth_of(i: int, guard: int = 0) -> int:
        if i in depths:
            return depths[i]
        if guard > len(skeleton.bones):
            return 0
        parent = skeleton.bones[i].parent
        d = 0 if parent < 0 or parent >= len(skeleton.bones) else depth_of(parent, guard + 1) + 1
        depths[i] = d
        return d

    return max((depth_of(i) for i in range(len(skeleton.bones))), default=0)


def _long_chains(skeleton: Skeleton, min_len: int = 3) -> list:
    """Unbranched bone chains, e.g. thigh->calf->foot."""
    chains = []
    for i in range(len(skeleton.bones)):
        if len(skeleton.children(i)) != 1:
            continue
        length, cur, guard = 1, i, 0
        while guard < len(skeleton.bones):
            kids = skeleton.children(cur)
            if len(kids) != 1:
                break
            cur = kids[0]
            length += 1
            guard += 1
        if length >= min_len:
            chains.append(length)
    return chains


def _has_symmetric_pairs(skeleton: Skeleton) -> bool:
    names = [b.name.lower() for b in skeleton.bones]
    pairs = 0
    for n in names:
        for a, b in (("left", "right"), ("_l", "_r"), (".l", ".r"), ("l_", "r_")):
            if a in n and n.replace(a, b) in names:
                pairs += 1
                break
    return pairs >= 4


def classify(model: ModelAsset, source_path: str = "") -> tuple:
    """Return (Classification, confidence, evidence dict). Structure over names."""
    evidence: dict = {}
    text = "%s %s" % (model.name, source_path)
    rig = analyse_skeleton(model.skeleton) if model.skeleton else RigAnalysis()
    evidence["rig"] = rig.to_dict()
    evidence["skinned"] = model.has_skinning
    evidence["animations"] = len(model.animations)
    evidence["materials"] = len(model.materials)
    evidence["textures"] = sum(len(m.textures) for m in model.materials)

    score = 0.0
    if model.has_skinning:
        score += 0.35
    if rig.is_humanoid:
        score += 0.35
    elif rig.bone_count >= 8:
        score += 0.15
    if model.animations:
        score += 0.1
    if evidence["textures"]:
        score += 0.05
    if CHAR_HINTS.search(text):
        score += 0.15
        evidence["name_hint"] = "character"

    if NPC_HINTS.search(text) and score >= 0.4:
        return Classification.NPC, min(0.97, score), evidence
    if CREATURE_HINTS.search(text) and (model.has_skinning or rig.bone_count >= 6):
        return Classification.CREATURE, min(0.95, score + 0.1), evidence
    if score >= 0.6:
        return Classification.CHARACTER, min(0.98, score), evidence
    if model.has_skinning and rig.bone_count >= 6:
        return Classification.CREATURE, min(0.8, score + 0.15), evidence

    if WEAPON_HINTS.search(text):
        return Classification.WEAPON, 0.7, evidence
    if ENV_HINTS.search(text) or model.triangle_count > 200000 and not model.has_skinning:
        return Classification.ENVIRONMENT, 0.55, evidence
    if PROP_HINTS.search(text):
        return Classification.PROP, 0.6, evidence
    if model.meshes:
        return Classification.PROP, 0.35, evidence
    return Classification.UNKNOWN, 0.2, evidence
