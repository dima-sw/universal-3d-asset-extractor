"""BVH motion-capture parser -> Skeleton + AnimationClip.

Registered as an animation parser; the skeleton it carries lets the asset graph
match it against a character rig by bone names.
"""
from __future__ import annotations

import math

from app.core.detector.base import FileContext
from app.core.types import (AnimationClip, AnimationTrack, Bone, Category, Keyframe, Skeleton)


def _euler_to_quat(order: list, angles: dict) -> tuple:
    q = (0.0, 0.0, 0.0, 1.0)
    axes = {"Xrotation": (1.0, 0.0, 0.0), "Yrotation": (0.0, 1.0, 0.0),
            "Zrotation": (0.0, 0.0, 1.0)}
    for channel in order:
        axis = axes.get(channel)
        if axis is None:
            continue
        half = math.radians(angles.get(channel, 0.0)) / 2.0
        s = math.sin(half)
        rot = (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))
        q = _mul(q, rot)
    return q


def _mul(a, b) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


class BvhParser:
    name = "bvh"
    priority = 70
    formats = ("BVH", "BVH (by extension)")

    def can_parse(self, ctx: FileContext, detection=None) -> bool:
        if ctx.ext == "bvh":
            return True
        return ctx.header[:9].upper().startswith(b"HIERARCHY")

    def detect_category(self) -> Category:
        return Category.ANIMATION

    def parse(self, ctx: FileContext) -> AnimationClip:
        text = ctx.path.read_text(encoding="utf-8", errors="ignore")
        tokens = text.split("\n")
        skeleton = Skeleton(name=ctx.path.stem)
        channels: list = []          # (bone_index, channel_name)
        stack: list = []
        i = 0
        n = len(tokens)

        while i < n:
            line = tokens[i].strip()
            i += 1
            if not line:
                continue
            parts = line.split()
            head = parts[0].upper()
            if head in ("ROOT", "JOINT"):
                parent = stack[-1] if stack else -1
                skeleton.bones.append(Bone(name=parts[1] if len(parts) > 1 else "bone",
                                           parent=parent))
                stack.append(len(skeleton.bones) - 1)
            elif head == "END":
                stack.append(stack[-1] if stack else -1)          # End Site: no bone
            elif head == "OFFSET" and stack and skeleton.bones:
                index = stack[-1]
                if 0 <= index < len(skeleton.bones):
                    try:
                        skeleton.bones[index].translation = tuple(float(x) for x in parts[1:4])
                    except ValueError:
                        pass
            elif head == "CHANNELS" and stack:
                index = stack[-1]
                for channel in parts[2:]:
                    channels.append((index, channel))
            elif line == "}":
                if stack:
                    stack.pop()
            elif head == "MOTION":
                break

        frames = 0
        frame_time = 1.0 / 30.0
        while i < n:
            line = tokens[i].strip()
            i += 1
            if line.lower().startswith("frames:"):
                frames = int(line.split(":")[1])
            elif line.lower().startswith("frame time:"):
                frame_time = float(line.split(":")[1])
                break

        clip = AnimationClip(name=ctx.path.stem, fps=1.0 / frame_time if frame_time else 30.0)
        tracks = {}
        for index, _channel in channels:
            if 0 <= index < len(skeleton.bones):
                tracks.setdefault(index, AnimationTrack(bone=skeleton.bones[index].name))

        frame_index = 0
        while i < n and frame_index < max(frames, 0):
            line = tokens[i].strip()
            i += 1
            if not line:
                continue
            try:
                values = [float(v) for v in line.split()]
            except ValueError:
                continue
            time = frame_index * frame_time
            positions: dict = {}
            rotations: dict = {}
            for (index, channel), value in zip(channels, values):
                if channel.endswith("position"):
                    positions.setdefault(index, {})[channel] = value
                elif channel.endswith("rotation"):
                    rotations.setdefault(index, {})[channel] = value
            for index, comps in positions.items():
                track = tracks.get(index)
                if track is None:
                    continue
                track.positions.append(Keyframe(time, (comps.get("Xposition", 0.0),
                                                       comps.get("Yposition", 0.0),
                                                       comps.get("Zposition", 0.0))))
            for index, comps in rotations.items():
                track = tracks.get(index)
                if track is None:
                    continue
                order = [c for _i, c in channels if _i == index and c.endswith("rotation")]
                track.rotations.append(Keyframe(time, _euler_to_quat(order, comps)))
            frame_index += 1

        clip.tracks = [t for t in tracks.values() if t.positions or t.rotations]
        clip.duration = max(0.0, (frame_index - 1) * frame_time)
        clip.skeleton_hint = skeleton.name
        clip.metadata["bones"] = len(skeleton.bones)
        clip.metadata["frames"] = frame_index
        return clip
