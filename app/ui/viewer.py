"""3D viewer widget.

Reads the internal representation (or anything the parsers can load, e.g. GLB)
and renders it with Qt's OpenGL. It knows nothing about parsers or containers:
give it a ModelAsset or a file path.

Supported: orbit / zoom / pan, solid / wireframe / textured, bone overlay,
animation playback with a timeline (CPU skinning).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QMatrix4x4, QSurfaceFormat, QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from app.core.types import AnimationClip, ModelAsset

VERTEX_SHADER = """
attribute highp vec3 a_position;
attribute mediump vec3 a_normal;
attribute mediump vec2 a_uv;
uniform highp mat4 u_mvp;
uniform highp mat4 u_model;
varying mediump vec3 v_normal;
varying mediump vec2 v_uv;
void main() {
    v_normal = mat3(u_model[0].xyz, u_model[1].xyz, u_model[2].xyz) * a_normal;
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""

FRAGMENT_SHADER = """
uniform mediump vec4 u_color;
uniform sampler2D u_texture;
uniform bool u_use_texture;
uniform bool u_flat;
varying mediump vec3 v_normal;
varying mediump vec2 v_uv;
void main() {
    mediump vec4 base = u_color;
    if (u_use_texture) base = texture2D(u_texture, v_uv) * u_color;
    if (u_flat) { gl_FragColor = base; return; }
    mediump vec3 n = normalize(v_normal);
    mediump float diff = max(dot(n, normalize(vec3(0.4, 0.8, 0.6))), 0.0);
    mediump float rim = 0.25 + 0.75 * diff;
    gl_FragColor = vec4(base.rgb * rim, base.a);
}
"""


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def _trs(translation, rotation, scale) -> np.ndarray:
    m = _quat_matrix(rotation)
    m[:3, :3] *= np.asarray(scale, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = translation
    return m


def _slerp(a, b, t: float):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:
        bx, by, bz, bw, dot = -bx, -by, -bz, -bw, -dot
    if dot > 0.9995:
        return (ax + (bx - ax) * t, ay + (by - ay) * t,
                az + (bz - az) * t, aw + (bw - aw) * t)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta) or 1e-6
    s0 = math.sin((1 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta
    return (ax * s0 + bx * s1, ay * s0 + by * s1, az * s0 + bz * s1, aw * s0 + bw * s1)


class _GpuMesh:
    """Expanded (non-indexed) draw data plus the unique arrays used for skinning."""

    def __init__(self):
        self.vbo: Optional[QOpenGLBuffer] = None
        self.wire_vbo: Optional[QOpenGLBuffer] = None
        self.vertex_count = 0            # expanded triangle vertices
        self.wire_count = 0
        self.texture: Optional[QOpenGLTexture] = None
        self.color = (0.78, 0.78, 0.82, 1.0)
        self.base_positions: Optional[np.ndarray] = None   # unique vertices
        self.normals: Optional[np.ndarray] = None
        self.uvs: Optional[np.ndarray] = None
        self.joints: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.wire_indices: Optional[np.ndarray] = None


class ModelViewer(QOpenGLWidget):
    """Public API: load_model, load_animation, set_camera, set_render_mode."""

    frame_changed = Signal(float)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)
        self.model: Optional[ModelAsset] = None
        self._meshes: list = []
        self._program: Optional[QOpenGLShaderProgram] = None
        self._bone_vbo: Optional[QOpenGLBuffer] = None
        self._bone_count = 0
        self._grid_vbo: Optional[QOpenGLBuffer] = None
        self._grid_count = 0
        self._ready = False
        self._dirty = False

        self.render_mode = "textured"       # solid | wireframe | textured
        self.show_bones = False
        self.show_grid = True

        self.yaw = 35.0
        self.pitch = 20.0
        self.distance = 3.0
        self.target = QVector3D(0, 0, 0)
        self._last_pos: Optional[QPoint] = None

        self.clip: Optional[AnimationClip] = None
        self.time = 0.0
        self.playing = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setMinimumSize(320, 240)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def load_model(self, model: ModelAsset) -> None:
        self.model = model
        self.clip = model.animations[0] if model and model.animations else None
        self.time = 0.0
        self._dirty = True
        self._frame_camera()
        self.status_changed.emit(
            "%s: %d verts, %d tris, %d bones, %d animations"
            % (model.name, model.vertex_count, model.triangle_count,
               len(model.skeleton.bones) if model.skeleton else 0, len(model.animations))
            if model else "no model")
        self.update()

    def load_file(self, path) -> bool:
        """Load any format the registered parsers understand (GLB, OBJ, ...)."""
        from app.core.detector import FileContext, identify
        from app.formats.models import parse_model
        ctx = FileContext(Path(path))
        try:
            model = parse_model(ctx, identify(ctx))
        except Exception as exc:
            self.status_changed.emit("cannot preview %s: %s" % (Path(path).name, exc))
            return False
        if model is None:
            self.status_changed.emit("no parser can preview %s" % Path(path).name)
            return False
        self.load_model(model)
        return True

    def load_animation(self, clip: Optional[AnimationClip]) -> None:
        self.clip = clip
        self.time = 0.0
        self.update()

    def set_camera(self, yaw: float = None, pitch: float = None, distance: float = None) -> None:
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = max(-89.0, min(89.0, pitch))
        if distance is not None:
            self.distance = max(0.05, distance)
        self.update()

    def set_render_mode(self, mode: str) -> None:
        self.render_mode = mode
        self.update()

    def play(self) -> None:
        if self.clip:
            self.playing = True
            self._timer.start()

    def pause(self) -> None:
        self.playing = False
        self._timer.stop()

    def reset_animation(self) -> None:
        self.time = 0.0
        self.frame_changed.emit(0.0)
        self.update()

    def set_time(self, t: float) -> None:
        self.time = max(0.0, t)
        self.update()

    # ------------------------------------------------------------------
    # GL
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        f = self.context().functions()
        f.glClearColor(0.11, 0.12, 0.14, 1.0)
        f.glEnable(0x0B71)                     # GL_DEPTH_TEST
        self._program = QOpenGLShaderProgram(self)
        self._program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER)
        self._program.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT_SHADER)
        self._program.bindAttributeLocation("a_position", 0)
        self._program.bindAttributeLocation("a_normal", 1)
        self._program.bindAttributeLocation("a_uv", 2)
        if not self._program.link():
            self.status_changed.emit("shader link failed: %s" % self._program.log())
            return
        self._ready = True
        self._build_grid()

    def resizeGL(self, w: int, h: int) -> None:
        self.context().functions().glViewport(0, 0, max(1, w), max(1, h))

    def paintGL(self) -> None:
        f = self.context().functions()
        f.glClear(0x00004000 | 0x00000100)      # COLOR | DEPTH
        if not self._ready or self._program is None:
            return
        if self._dirty:
            self._upload()
            self._dirty = False
        if not self._meshes and not self._grid_count:
            return

        proj = QMatrix4x4()
        proj.perspective(45.0, self.width() / max(1, self.height()), 0.01,
                         max(100.0, self.distance * 10))
        view = QMatrix4x4()
        eye = self._eye()
        view.lookAt(eye, self.target, QVector3D(0, 1, 0))
        model_matrix = QMatrix4x4()
        mvp = proj * view * model_matrix

        self._program.bind()
        self._program.setUniformValue("u_mvp", mvp)
        self._program.setUniformValue("u_model", model_matrix)

        if self.show_grid and self._grid_vbo is not None:
            self._draw_lines(self._grid_vbo, self._grid_count, (0.25, 0.26, 0.3, 1.0))

        skin = self._skin_matrices() if (self.clip and self.model and self.model.skeleton) else None
        for mesh in self._meshes:
            if skin is not None and mesh.joints is not None:
                self._apply_skin(mesh, skin)
            self._draw_mesh(mesh)

        if self.show_bones and self.model and self.model.skeleton:
            self._draw_bones(skin)
        self._program.release()

    # -- drawing helpers ---------------------------------------------------
    def _draw_mesh(self, mesh: _GpuMesh) -> None:
        f = self.context().functions()
        if mesh.vbo is None:
            return
        mesh.vbo.bind()
        stride = 8 * 4
        self._program.enableAttributeArray(0)
        self._program.setAttributeBuffer(0, 0x1406, 0, 3, stride)          # GL_FLOAT
        self._program.enableAttributeArray(1)
        self._program.setAttributeBuffer(1, 0x1406, 3 * 4, 3, stride)
        self._program.enableAttributeArray(2)
        self._program.setAttributeBuffer(2, 0x1406, 6 * 4, 2, stride)

        use_texture = self.render_mode == "textured" and mesh.texture is not None
        self._program.setUniformValue("u_use_texture", use_texture)
        self._program.setUniformValue("u_flat", False)
        self._set_color(mesh.color)
        if use_texture:
            mesh.texture.bind(0)
            self._program.setUniformValue("u_texture", 0)

        # Vertices are expanded at upload time: PySide6's glDrawElements binding
        # has no way to express a zero (start of buffer) index offset.
        if self.render_mode == "wireframe":
            if mesh.wire_vbo is not None and mesh.wire_count:
                mesh.vbo.release()
                self._set_color((0.6, 0.85, 1.0, 1.0))
                self._program.setUniformValue("u_use_texture", False)
                self._program.setUniformValue("u_flat", True)
                self._draw_lines(mesh.wire_vbo, mesh.wire_count, (0.6, 0.85, 1.0, 1.0))
                return
        elif mesh.vertex_count:
            f.glDrawArrays(0x0004, 0, mesh.vertex_count)               # GL_TRIANGLES

        if use_texture:
            mesh.texture.release(0)
        mesh.vbo.release()

    def _draw_lines(self, vbo: QOpenGLBuffer, count: int, color) -> None:
        f = self.context().functions()
        vbo.bind()
        self._program.enableAttributeArray(0)
        self._program.setAttributeBuffer(0, 0x1406, 0, 3, 3 * 4)
        self._program.disableAttributeArray(1)
        self._program.disableAttributeArray(2)
        self._program.setUniformValue("u_use_texture", False)
        self._program.setUniformValue("u_flat", True)
        self._set_color(color)
        f.glDrawArrays(0x0001, 0, count)
        vbo.release()

    def _set_color(self, color) -> None:
        self._program.setUniformValue("u_color", float(color[0]), float(color[1]),
                                      float(color[2]), float(color[3]))

    def _draw_bones(self, skin) -> None:
        skeleton = self.model.skeleton
        world = self._world_matrices(skin is not None)
        points = []
        for i, bone in enumerate(skeleton.bones):
            if bone.parent < 0 or bone.parent >= len(skeleton.bones):
                continue
            a = world[bone.parent][:3, 3]
            b = world[i][:3, 3]
            points.extend([a[0], a[1], a[2], b[0], b[1], b[2]])
        if not points:
            return
        data = np.asarray(points, dtype=np.float32)
        if self._bone_vbo is None:
            self._bone_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._bone_vbo.create()
        self._bone_vbo.bind()
        self._bone_vbo.allocate(data.tobytes(), data.nbytes)
        self._bone_vbo.release()
        self._draw_lines(self._bone_vbo, len(data) // 3, (1.0, 0.7, 0.2, 1.0))

    # -- data upload -------------------------------------------------------
    def _upload(self) -> None:
        for mesh in self._meshes:
            for buf in (mesh.vbo, mesh.wire_vbo):
                if buf is not None:
                    buf.destroy()
            if mesh.texture is not None:
                mesh.texture.destroy()
        self._meshes = []
        if not self.model:
            return

        for src in self.model.meshes:
            if not src.vertices:
                continue
            gpu = _GpuMesh()
            count = len(src.vertices)
            positions = np.asarray(src.vertices, dtype=np.float32).reshape(-1, 3)
            normals = (np.asarray(src.normals, dtype=np.float32).reshape(-1, 3)
                       if len(src.normals) == count else np.tile(np.array([[0, 0, 1]],
                                                                          dtype=np.float32),
                                                                 (count, 1)))
            uvs = (np.asarray(src.uv_channels[0], dtype=np.float32).reshape(-1, 2)
                   if src.uv_channels and len(src.uv_channels[0]) == count
                   else np.zeros((count, 2), dtype=np.float32))
            gpu.base_positions = positions
            gpu.normals = normals
            gpu.uvs = uvs
            if src.is_skinned and len(src.joints) == count:
                gpu.joints = np.asarray(src.joints, dtype=np.int32).reshape(-1, 4)
                gpu.weights = np.asarray(src.weights, dtype=np.float32).reshape(-1, 4)

            indices = np.asarray(src.indices or list(range(count)), dtype=np.int64)
            indices = indices[(indices >= 0) & (indices < count)]
            indices = indices[:len(indices) // 3 * 3]
            gpu.indices = indices

            interleaved = np.hstack([positions[indices], normals[indices],
                                     uvs[indices]]).astype(np.float32)
            gpu.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            gpu.vbo.create()
            gpu.vbo.bind()
            gpu.vbo.allocate(interleaved.tobytes(), interleaved.nbytes)
            gpu.vbo.release()
            gpu.vertex_count = len(indices)

            tris = indices.reshape(-1, 3)
            edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]).reshape(-1)
            gpu.wire_indices = edges
            wire = positions[edges].astype(np.float32)
            gpu.wire_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            gpu.wire_vbo.create()
            gpu.wire_vbo.bind()
            gpu.wire_vbo.allocate(wire.tobytes(), wire.nbytes)
            gpu.wire_vbo.release()
            gpu.wire_count = len(edges)

            if 0 <= src.material < len(self.model.materials):
                material = self.model.materials[src.material]
                gpu.color = tuple(material.base_color)
                gpu.texture = self._make_texture(material)
            self._meshes.append(gpu)

    def _make_texture(self, material) -> Optional[QOpenGLTexture]:
        tex = material.textures.get("diffuse")
        if tex is None:
            return None
        image = QImage()
        if tex.data:
            image.loadFromData(tex.data)
        elif tex.path and Path(tex.path).exists():
            image.load(str(tex.path))
        if image.isNull():
            return None
        try:
            gl_tex = QOpenGLTexture(image.mirrored())
            gl_tex.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
            gl_tex.setMagnificationFilter(QOpenGLTexture.Linear)
            gl_tex.setWrapMode(QOpenGLTexture.Repeat)
            return gl_tex
        except Exception:
            return None

    def _build_grid(self, size: float = 2.0, step: float = 0.25) -> None:
        points = []
        n = int(size / step)
        for i in range(-n, n + 1):
            x = i * step
            points += [x, 0.0, -size, x, 0.0, size, -size, 0.0, x, size, 0.0, x]
        data = np.asarray(points, dtype=np.float32)
        self._grid_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._grid_vbo.create()
        self._grid_vbo.bind()
        self._grid_vbo.allocate(data.tobytes(), data.nbytes)
        self._grid_vbo.release()
        self._grid_count = len(data) // 3

    # -- animation ---------------------------------------------------------
    def _tick(self) -> None:
        if not self.clip or not self.playing:
            return
        self.time += 0.016
        if self.clip.duration > 0 and self.time > self.clip.duration:
            self.time = 0.0
        self.frame_changed.emit(self.time)
        self.update()

    def _sample(self, keys, default):
        if not keys:
            return None
        if len(keys) == 1 or self.time <= keys[0].time:
            return keys[0].value
        if self.time >= keys[-1].time:
            return keys[-1].value
        for i in range(1, len(keys)):
            if keys[i].time >= self.time:
                a, b = keys[i - 1], keys[i]
                span = (b.time - a.time) or 1e-6
                t = (self.time - a.time) / span
                if len(a.value) == 4:
                    return _slerp(a.value, b.value, t)
                return tuple(av + (bv - av) * t for av, bv in zip(a.value, b.value))
        return default

    def _world_matrices(self, animated: bool = True) -> list:
        skeleton = self.model.skeleton
        tracks = {t.bone: t for t in (self.clip.tracks if (self.clip and animated) else [])}
        world = []
        for i, bone in enumerate(skeleton.bones):
            track = tracks.get(bone.name)
            translation = bone.translation
            rotation = bone.rotation
            scale = bone.scale
            if track is not None:
                translation = self._sample(track.positions, translation) or translation
                rotation = self._sample(track.rotations, rotation) or rotation
                scale = self._sample(track.scales, scale) or scale
            local = _trs(translation, rotation, scale)
            if 0 <= bone.parent < len(world):
                world.append(world[bone.parent] @ local)
            else:
                world.append(local)
        return world

    def _skin_matrices(self):
        from app.exporters.gltf import inverse_bind_matrices
        skeleton = self.model.skeleton
        world = self._world_matrices(True)
        inv = inverse_bind_matrices(skeleton)
        out = []
        for i in range(len(skeleton.bones)):
            ibm = np.asarray(inv[i], dtype=np.float32).reshape(4, 4).T   # stored column-major
            out.append(world[i] @ ibm)
        return out

    def _apply_skin(self, mesh: _GpuMesh, skin: list) -> None:
        positions = mesh.base_positions
        joints = mesh.joints
        weights = mesh.weights
        if positions is None or joints is None or weights is None or not skin:
            return
        mats = np.asarray(skin, dtype=np.float32)
        n = positions.shape[0]
        homo = np.hstack([positions, np.ones((n, 1), dtype=np.float32)])
        out = np.zeros((n, 3), dtype=np.float32)
        for slot in range(joints.shape[1]):
            idx = np.clip(joints[:, slot], 0, len(mats) - 1)
            w = weights[:, slot][:, None]
            transformed = np.einsum("nij,nj->ni", mats[idx], homo)[:, :3]
            out += transformed * w
        idx = mesh.indices
        if idx is None or not len(idx):
            return
        interleaved = np.hstack([out[idx], mesh.normals[idx],
                                 mesh.uvs[idx]]).astype(np.float32)
        mesh.vbo.bind()
        mesh.vbo.write(0, interleaved.tobytes(), interleaved.nbytes)
        mesh.vbo.release()
        if mesh.wire_vbo is not None and mesh.wire_indices is not None:
            wire = out[mesh.wire_indices].astype(np.float32)
            mesh.wire_vbo.bind()
            mesh.wire_vbo.write(0, wire.tobytes(), wire.nbytes)
            mesh.wire_vbo.release()

    # -- camera ------------------------------------------------------------
    def _eye(self) -> QVector3D:
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        x = self.distance * math.cos(rp) * math.sin(ry)
        y = self.distance * math.sin(rp)
        z = self.distance * math.cos(rp) * math.cos(ry)
        return QVector3D(self.target.x() + x, self.target.y() + y, self.target.z() + z)

    def _frame_camera(self) -> None:
        if not self.model or not self.model.meshes:
            return
        points = [v for mesh in self.model.meshes for v in mesh.vertices[:5000]]
        if not points:
            return
        arr = np.asarray(points, dtype=np.float32)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0
        self.target = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.distance = radius * 3.0

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        if event.buttons() & Qt.LeftButton:
            self.yaw -= dx * 0.5
            self.pitch = max(-89.0, min(89.0, self.pitch + dy * 0.5))
        elif event.buttons() & (Qt.MiddleButton | Qt.RightButton):
            scale = self.distance * 0.002
            right = QVector3D.crossProduct(self.target - self._eye(), QVector3D(0, 1, 0)).normalized()
            self.target += right * (dx * scale) + QVector3D(0, 1, 0) * (dy * scale)
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        self.distance = max(0.05, self.distance * (0.9 ** delta))
        self.update()
