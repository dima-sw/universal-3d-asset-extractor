"""Format-independent internal asset representation.

Every parser produces these types; every exporter consumes them.
A parser must never talk to an exporter directly.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AssetType(str, Enum):
    MODEL = "model"
    MESH = "mesh"
    SKELETON = "skeleton"
    ANIMATION = "animation"
    TEXTURE = "texture"
    MATERIAL = "material"
    AUDIO = "audio"
    CONTAINER = "container"
    CHARACTER = "character"
    UNKNOWN = "unknown"


class Category(str, Enum):
    ARCHIVE = "archive"
    DISC_IMAGE = "disc_image"
    GAME_CONTAINER = "game_container"
    EXECUTABLE = "executable"
    MODEL = "model"
    TEXTURE = "texture"
    ANIMATION = "animation"
    AUDIO = "audio"
    COMPRESSED = "compressed"
    UNKNOWN = "unknown"
    OTHER = "other"


class Classification(str, Enum):
    CHARACTER = "Character"
    CREATURE = "Creature"
    NPC = "NPC"
    PROP = "Prop"
    WEAPON = "Weapon"
    ENVIRONMENT = "Environment"
    UNKNOWN = "Unknown"


@dataclass
class DetectionResult:
    detected: bool = False
    format_name: str = "Unknown"
    confidence: float = 0.0
    category: Category = Category.UNKNOWN
    metadata: dict = field(default_factory=dict)
    detector: str = ""

    @property
    def engine(self) -> Optional[str]:
        return self.metadata.get("engine")

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "format_name": self.format_name,
            "confidence": round(self.confidence, 4),
            "category": self.category.value,
            "metadata": self.metadata,
            "detector": self.detector,
        }


def not_detected() -> DetectionResult:
    return DetectionResult()


# --------------------------------------------------------------------------
# Geometry / rig
# --------------------------------------------------------------------------
@dataclass
class Bone:
    name: str
    parent: int = -1                          # index into Skeleton.bones, -1 = root
    translation: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)    # xyzw quaternion
    scale: tuple = (1.0, 1.0, 1.0)
    inverse_bind: Optional[list] = None        # 16 floats, column-major


@dataclass
class Skeleton:
    name: str = "Skeleton"
    bones: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def index_of(self, name: str) -> int:
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1

    def roots(self) -> list:
        return [i for i, b in enumerate(self.bones) if b.parent < 0]

    def children(self, index: int) -> list:
        return [i for i, b in enumerate(self.bones) if b.parent == index]

    def validate(self) -> list:
        """Return a list of structural problems (bad parents, cycles)."""
        problems = []
        n = len(self.bones)
        for i, b in enumerate(self.bones):
            if b.parent >= n or b.parent == i:
                problems.append("bone %d (%s) invalid parent %d" % (i, b.name, b.parent))
        for i in range(n):
            seen, cur, steps = set(), i, 0
            while cur >= 0 and steps <= n:
                if cur in seen:
                    problems.append("cycle at bone %d (%s)" % (i, self.bones[i].name))
                    break
                seen.add(cur)
                p = self.bones[cur].parent
                cur = p if 0 <= p < n else -1
                steps += 1
        return problems


@dataclass
class MorphTarget:
    name: str
    positions: list = field(default_factory=list)    # list[(dx,dy,dz)]
    normals: list = field(default_factory=list)


@dataclass
class Mesh:
    name: str = "Mesh"
    vertices: list = field(default_factory=list)      # list[(x,y,z)]
    normals: list = field(default_factory=list)
    uv_channels: list = field(default_factory=list)   # list[list[(u,v)]]
    colors: list = field(default_factory=list)
    indices: list = field(default_factory=list)       # flat triangle list
    material: int = -1                                 # index into ModelAsset.materials
    joints: list = field(default_factory=list)        # list[(j0,j1,j2,j3)]
    weights: list = field(default_factory=list)       # list[(w0,w1,w2,w3)]
    morph_targets: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_skinned(self) -> bool:
        return bool(self.joints) and bool(self.weights)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


@dataclass
class TextureAsset:
    name: str = "Texture"
    width: int = 0
    height: int = 0
    channels: int = 0
    bit_depth: int = 8
    has_alpha: bool = False
    color_space: str = "srgb"
    source_format: str = ""
    data: Optional[bytes] = None      # encoded bytes when held in memory
    path: Optional[str] = None        # on-disk path when materialised
    usage: str = "diffuse"            # diffuse|normal|roughness|metallic|ao|emission|mask|specular
    metadata: dict = field(default_factory=dict)


@dataclass
class Material:
    name: str = "Material"
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.8
    double_sided: bool = False
    textures: dict = field(default_factory=dict)   # usage -> TextureAsset
    metadata: dict = field(default_factory=dict)


@dataclass
class Keyframe:
    time: float
    value: tuple


@dataclass
class AnimationTrack:
    bone: str
    positions: list = field(default_factory=list)   # list[Keyframe]
    rotations: list = field(default_factory=list)
    scales: list = field(default_factory=list)


@dataclass
class AnimationClip:
    name: str = "Animation"
    duration: float = 0.0
    fps: float = 30.0
    tracks: list = field(default_factory=list)
    skeleton_hint: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelAsset:
    name: str = "Model"
    meshes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    skeleton: Optional[Skeleton] = None
    animations: list = field(default_factory=list)
    source_format: str = ""
    coordinate_system: str = "y_up_rh"
    unit_scale: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return sum(len(m.vertices) for m in self.meshes)

    @property
    def triangle_count(self) -> int:
        return sum(m.triangle_count for m in self.meshes)

    @property
    def has_skinning(self) -> bool:
        return any(m.is_skinned for m in self.meshes)


# --------------------------------------------------------------------------
# Graph-level asset record (what the DB, report and UI see)
# --------------------------------------------------------------------------
@dataclass
class Asset:
    name: str
    type: AssetType
    source: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    confidence: float = 0.0
    classification: Classification = Classification.UNKNOWN
    format_name: str = ""
    engine: Optional[str] = None
    payload: Any = None                  # ModelAsset / TextureAsset / AnimationClip / ...
    content_hash: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "classification": self.classification.value,
            "format": self.format_name,
            "engine": self.engine,
            "hash": self.content_hash,
            "metadata": self.metadata,
        }


class Relation(str, Enum):
    MESH_MATERIAL = "mesh->material"
    MATERIAL_TEXTURE = "material->texture"
    MESH_SKELETON = "mesh->skeleton"
    ANIMATION_SKELETON = "animation->skeleton"
    CHARACTER_MESH = "character->mesh"
    CHARACTER_SKELETON = "character->skeleton"
    CHARACTER_ANIMATION = "character->animation"
    CHARACTER_TEXTURE = "character->texture"
    CONTAINS = "contains"
    DUPLICATE_OF = "duplicate-of"


@dataclass
class AssetReference:
    source_asset: str
    target_asset: str
    relation: Relation
    confidence: float = 1.0
