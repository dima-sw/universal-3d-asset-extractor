"""Validation + confidence scoring for parsed assets."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.types import DetectionResult, ModelAsset


@dataclass
class Validation:
    valid: bool = True
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    quality: float = 0.0

    def to_dict(self) -> dict:
        return {"valid": self.valid, "quality": round(self.quality, 3),
                "issues": self.issues, "warnings": self.warnings}


def validate_model(model: ModelAsset) -> Validation:
    v = Validation()
    if not model.meshes:
        v.valid = False
        v.issues.append("no mesh in model")
        return v

    total_vertices = 0
    for mesh in model.meshes:
        n = len(mesh.vertices)
        total_vertices += n
        if n == 0:
            v.warnings.append("mesh %s has no vertices" % mesh.name)
            continue
        if mesh.indices:
            if len(mesh.indices) % 3:
                v.warnings.append("mesh %s index count not a multiple of 3" % mesh.name)
            bad = [i for i in mesh.indices[:20000] if i < 0 or i >= n]
            if bad:
                v.valid = False
                v.issues.append("mesh %s has %d out-of-range indices" % (mesh.name, len(bad)))
        if mesh.normals and len(mesh.normals) != n:
            v.warnings.append("mesh %s normal count mismatch" % mesh.name)
        for ch in mesh.uv_channels:
            if len(ch) != n:
                v.warnings.append("mesh %s uv count mismatch" % mesh.name)
                break
        if mesh.is_skinned:
            if len(mesh.joints) != n or len(mesh.weights) != n:
                v.warnings.append("mesh %s skin arrays mismatch" % mesh.name)
            elif model.skeleton:
                max_joint = max((max(j) for j in mesh.joints[:20000] if j), default=0)
                if max_joint >= len(model.skeleton.bones):
                    v.valid = False
                    v.issues.append("mesh %s references bone %d, skeleton has %d"
                                    % (mesh.name, max_joint, len(model.skeleton.bones)))
        for morph in mesh.morph_targets:
            if morph.positions and len(morph.positions) != n:
                v.warnings.append("morph target %s vertex mismatch" % morph.name)

    if model.skeleton:
        problems = model.skeleton.validate()
        if problems:
            v.valid = False
            v.issues.extend(problems[:5])

    for clip in model.animations:
        if not clip.tracks:
            v.warnings.append("animation %s has no tracks" % clip.name)
        elif clip.duration <= 0:
            v.warnings.append("animation %s has zero duration" % clip.name)

    v.quality = _quality(model, v, total_vertices)
    return v


def _quality(model: ModelAsset, v: Validation, total_vertices: int) -> float:
    score = 0.0
    if total_vertices > 0:
        score += 0.35
    if any(m.indices for m in model.meshes):
        score += 0.15
    if any(m.normals for m in model.meshes):
        score += 0.05
    if any(m.uv_channels for m in model.meshes):
        score += 0.1
    if model.materials:
        score += 0.05
    if any(mat.textures for mat in model.materials):
        score += 0.1
    if model.skeleton and model.skeleton.bones:
        score += 0.1
    if model.animations:
        score += 0.1
    score -= 0.25 * len(v.issues)
    score -= 0.03 * len(v.warnings)
    return max(0.0, min(1.0, score))


def confidence(detection: DetectionResult, validation: Validation,
               parser_specific: bool = True, errors: int = 0) -> float:
    """Combine detection strength, parse quality and error count."""
    base = 0.45 * detection.confidence + 0.45 * validation.quality
    if parser_specific:
        base += 0.1
    if not validation.valid:
        base *= 0.5
    base -= 0.05 * min(errors, 5)
    return max(0.05, min(0.99, base))
