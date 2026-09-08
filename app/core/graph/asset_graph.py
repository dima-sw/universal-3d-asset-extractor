"""Asset relationship graph + character reconstruction.

Relations come from explicit references when the format provides them (GUID,
path, id) and from evidence otherwise: shared directory, name stem, skeleton
bone-name overlap, texture naming.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Iterable, Optional

from app.core.graph.character import analyse_skeleton
from app.core.types import (Asset, AssetReference, AssetType, Classification, ModelAsset,
                            Relation)

SPLIT = re.compile(r"[^A-Za-z0-9]+")
NOISE = {"mesh", "model", "skin", "skinned", "lod0", "lod1", "lod", "sk", "sm", "char",
         "character", "body", "final", "export", "anim", "animation", "clip", "tex",
         "texture", "diffuse", "albedo", "normal", "base", "color", "d", "n", "s"}


def tokens(name: str) -> set:
    parts = [p.lower() for p in SPLIT.split(PurePath(name).stem) if p]
    return {p for p in parts if p and p not in NOISE and not p.isdigit()}


def name_similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard > 0:
            return min(1.0, jaccard + 0.1)
    return difflib.SequenceMatcher(None, PurePath(a).stem.lower(),
                                   PurePath(b).stem.lower()).ratio() * 0.8


def path_proximity(a: str, b: str) -> float:
    pa, pb = PurePath(a).parts[:-1], PurePath(b).parts[:-1]
    if not pa or not pb:
        return 0.0
    common = 0
    for x, y in zip(pa, pb):
        if x.lower() != y.lower():
            break
        common += 1
    return common / max(len(pa), len(pb))


@dataclass
class CharacterGroup:
    name: str
    model_asset: Asset
    animations: list = field(default_factory=list)     # list[Asset]
    textures: list = field(default_factory=list)
    skeleton_asset: Optional[Asset] = None
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "model": self.model_asset.id,
                "animations": [a.name for a in self.animations],
                "textures": [t.name for t in self.textures],
                "confidence": round(self.confidence, 3), "evidence": self.evidence}


class AssetGraph:
    def __init__(self):
        self.assets: dict = {}
        self.references: list = []
        self._by_type: dict = {}
        self._by_hash: dict = {}

    # -- population --------------------------------------------------------
    def add(self, asset: Asset) -> Asset:
        """Add an asset. Identical content is recorded as a duplicate edge."""
        if asset.content_hash and asset.content_hash in self._by_hash:
            original = self._by_hash[asset.content_hash]
            if original.id != asset.id:
                asset.metadata["duplicate_of"] = original.id
                self.assets[asset.id] = asset
                self._by_type.setdefault(asset.type, []).append(asset)
                self.link(asset.id, original.id, Relation.DUPLICATE_OF)
                return asset
        self.assets[asset.id] = asset
        self._by_type.setdefault(asset.type, []).append(asset)
        if asset.content_hash:
            self._by_hash.setdefault(asset.content_hash, asset)
        return asset

    def link(self, source_id: str, target_id: str, relation: Relation,
             confidence: float = 1.0) -> AssetReference:
        ref = AssetReference(source_id, target_id, relation, confidence)
        self.references.append(ref)
        return ref

    # -- queries -----------------------------------------------------------
    def by_type(self, type_: AssetType) -> list:
        return list(self._by_type.get(type_, []))

    def duplicates(self) -> list:
        return [a for a in self.assets.values() if "duplicate_of" in a.metadata]

    def neighbours(self, asset_id: str, relation: Optional[Relation] = None) -> list:
        out = []
        for ref in self.references:
            if ref.source_asset == asset_id and (relation is None or ref.relation == relation):
                out.append(self.assets.get(ref.target_asset))
            elif ref.target_asset == asset_id and (relation is None or ref.relation == relation):
                out.append(self.assets.get(ref.source_asset))
        return [a for a in out if a is not None]

    def subgraph(self, asset_id: str, depth: int = 2) -> dict:
        seen = {asset_id}
        frontier = [asset_id]
        for _ in range(depth):
            nxt = []
            for aid in frontier:
                for neighbour in self.neighbours(aid):
                    if neighbour.id not in seen:
                        seen.add(neighbour.id)
                        nxt.append(neighbour.id)
            frontier = nxt
        return {aid: self.assets[aid] for aid in seen if aid in self.assets}

    # -- reconstruction ----------------------------------------------------
    def reconstruct_characters(self, min_confidence: float = 0.45) -> list:
        """Group model + animations + textures that belong to the same subject."""
        groups: list = []
        models = [a for a in self.by_type(AssetType.MODEL)
                  if "duplicate_of" not in a.metadata]
        animations = [a for a in self.by_type(AssetType.ANIMATION)
                      if "duplicate_of" not in a.metadata]
        textures = [a for a in self.by_type(AssetType.TEXTURE)
                    if "duplicate_of" not in a.metadata]

        for asset in models:
            model: ModelAsset = asset.payload
            if not isinstance(model, ModelAsset):
                continue
            if asset.classification not in (Classification.CHARACTER, Classification.CREATURE,
                                            Classification.NPC):
                continue
            group = CharacterGroup(name=asset.name, model_asset=asset,
                                   confidence=asset.confidence)
            bone_names = {b.name.lower() for b in (model.skeleton.bones if model.skeleton else [])}

            # animations already carried by the model file
            group.evidence["embedded_animations"] = len(model.animations)

            for anim_asset in animations:
                score, why = self._animation_match(asset, anim_asset, bone_names)
                if score >= 0.5:
                    group.animations.append(anim_asset)
                    self.link(asset.id, anim_asset.id, Relation.CHARACTER_ANIMATION, score)
                    group.evidence.setdefault("animation_matches", {})[anim_asset.name] = why

            for tex_asset in textures:
                score = self._texture_match(asset, tex_asset, model)
                if score >= 0.55:
                    group.textures.append(tex_asset)
                    self.link(asset.id, tex_asset.id, Relation.CHARACTER_TEXTURE, score)

            rig = analyse_skeleton(model.skeleton) if model.skeleton else None
            if rig:
                group.evidence["rig"] = rig.to_dict()
            group.confidence = min(0.99, asset.confidence
                                   + (0.05 if group.animations else 0.0)
                                   + (0.03 if group.textures else 0.0))
            if group.confidence >= min_confidence:
                groups.append(group)
        return groups

    def _animation_match(self, model_asset: Asset, anim_asset: Asset, bone_names: set) -> tuple:
        clip = anim_asset.payload
        why = {}
        score = 0.0
        track_names = set()
        if clip is not None and getattr(clip, "tracks", None):
            track_names = {t.bone.lower() for t in clip.tracks}
        if bone_names and track_names:
            overlap = len(bone_names & track_names) / max(1, len(track_names))
            why["bone_overlap"] = round(overlap, 3)
            score += overlap * 0.8
        prox = path_proximity(model_asset.source, anim_asset.source)
        why["path_proximity"] = round(prox, 3)
        score += prox * 0.3
        sim = name_similarity(model_asset.name, anim_asset.name)
        why["name_similarity"] = round(sim, 3)
        score += sim * 0.3
        skeleton_hint = getattr(clip, "skeleton_hint", None)
        if skeleton_hint and model_asset.metadata.get("skeleton_name") == skeleton_hint:
            score += 0.5
            why["skeleton_hint"] = skeleton_hint
        return min(1.0, score), why

    def _texture_match(self, model_asset: Asset, tex_asset: Asset, model: ModelAsset) -> float:
        for mat in model.materials:
            for tex in mat.textures.values():
                if tex.path and tex.path == tex_asset.metadata.get("path"):
                    return 1.0
                if tex.name and tex.name.lower() == tex_asset.name.lower():
                    return 0.95
        score = name_similarity(model_asset.name, tex_asset.name) * 0.7
        score += path_proximity(model_asset.source, tex_asset.source) * 0.45
        return min(1.0, score)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "assets": [a.to_dict() for a in self.assets.values()],
            "references": [{"source": r.source_asset, "target": r.target_asset,
                            "relation": r.relation.value,
                            "confidence": round(r.confidence, 3)} for r in self.references],
        }

    def counts(self) -> dict:
        return {t.value: len(v) for t, v in self._by_type.items()}

    def extend(self, assets: Iterable[Asset]) -> None:
        for a in assets:
            self.add(a)
