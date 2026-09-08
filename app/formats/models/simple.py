"""STL and PLY parsers (ascii + binary)."""
from __future__ import annotations

import struct

from app.core.detector.base import FileContext
from app.core.types import Mesh, ModelAsset
from app.formats.models.base import ModelParser, ParseError


class StlParser(ModelParser):
    name = "stl"
    priority = 70
    formats = ("STL (ascii)", "STL (binary)", "STL (by extension)")

    def can_parse(self, ctx: FileContext, detection=None) -> bool:
        if detection is not None and detection.format_name in self.formats:
            return True
        return ctx.ext == "stl"

    def parse(self, ctx: FileContext) -> ModelAsset:
        data_head = ctx.header[:512]
        is_ascii = data_head.lstrip()[:6].lower() == b"solid " and b"facet" in ctx.header[:4096].lower()
        mesh = Mesh(name=ctx.path.stem)
        if is_ascii:
            self._parse_ascii(ctx, mesh)
        else:
            self._parse_binary(ctx, mesh)
        model = ModelAsset(name=ctx.path.stem, source_format="STL")
        model.meshes.append(mesh)
        return model

    def _parse_ascii(self, ctx: FileContext, mesh: Mesh) -> None:
        normal = (0.0, 0.0, 1.0)
        with open(ctx.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "facet" and len(parts) >= 5:
                    normal = tuple(float(x) for x in parts[2:5])
                elif parts[0] == "vertex" and len(parts) >= 4:
                    mesh.vertices.append(tuple(float(x) for x in parts[1:4]))
                    mesh.normals.append(normal)
                    mesh.indices.append(len(mesh.vertices) - 1)

    def _parse_binary(self, ctx: FileContext, mesh: Mesh) -> None:
        with open(ctx.path, "rb") as f:
            f.seek(80)
            raw = f.read(4)
            if len(raw) < 4:
                raise ParseError("truncated STL header")
            count = struct.unpack("<I", raw)[0]
            expected = 84 + count * 50
            if expected > ctx.size + 1024:
                raise ParseError("STL triangle count %d does not match file size" % count)
            for _ in range(count):
                chunk = f.read(50)
                if len(chunk) < 50:
                    break
                nx, ny, nz = struct.unpack_from("<3f", chunk, 0)
                for v in range(3):
                    x, y, z = struct.unpack_from("<3f", chunk, 12 + v * 12)
                    mesh.vertices.append((x, y, z))
                    mesh.normals.append((nx, ny, nz))
                    mesh.indices.append(len(mesh.vertices) - 1)


PLY_TYPES = {
    "char": ("b", 1), "uchar": ("B", 1), "int8": ("b", 1), "uint8": ("B", 1),
    "short": ("h", 2), "ushort": ("H", 2), "int16": ("h", 2), "uint16": ("H", 2),
    "int": ("i", 4), "uint": ("I", 4), "int32": ("i", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
}


class PlyParser(ModelParser):
    name = "ply"
    priority = 70
    formats = ("PLY", "PLY (ascii)", "PLY (by extension)")

    def can_parse(self, ctx: FileContext, detection=None) -> bool:
        if detection is not None and detection.format_name in self.formats:
            return True
        return ctx.ext == "ply"

    def parse(self, ctx: FileContext) -> ModelAsset:
        with open(ctx.path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    raise ParseError("unterminated PLY header")
                text = line.decode("ascii", "ignore").strip()
                header_lines.append(text)
                if text == "end_header":
                    break
            fmt, elements = self._parse_header(header_lines)
            mesh = Mesh(name=ctx.path.stem)
            if fmt == "ascii":
                self._read_ascii(f, elements, mesh)
            else:
                self._read_binary(f, elements, mesh, big_endian=(fmt == "binary_big_endian"))
        model = ModelAsset(name=ctx.path.stem, source_format="PLY")
        model.meshes.append(mesh)
        return model

    def _parse_header(self, lines) -> tuple:
        fmt = "ascii"
        elements: list = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
            elif parts[0] == "property" and elements:
                if parts[1] == "list":
                    elements[-1]["props"].append(("list", parts[2], parts[3], parts[4]))
                else:
                    elements[-1]["props"].append(("scalar", parts[1], parts[2]))
        return fmt, elements

    def _read_ascii(self, f, elements, mesh: Mesh) -> None:
        for el in elements:
            for _ in range(el["count"]):
                line = f.readline().decode("ascii", "ignore").split()
                if not line:
                    break
                if el["name"] == "vertex":
                    values = {}
                    for i, prop in enumerate(el["props"]):
                        if prop[0] == "scalar" and i < len(line):
                            values[prop[2]] = float(line[i])
                    mesh.vertices.append((values.get("x", 0.0), values.get("y", 0.0),
                                          values.get("z", 0.0)))
                    if "nx" in values:
                        mesh.normals.append((values["nx"], values.get("ny", 0.0),
                                             values.get("nz", 0.0)))
                    if "s" in values or "u" in values:
                        if not mesh.uv_channels:
                            mesh.uv_channels.append([])
                        mesh.uv_channels[0].append((values.get("s", values.get("u", 0.0)),
                                                    values.get("t", values.get("v", 0.0))))
                elif el["name"] == "face" and line:
                    n = int(line[0])
                    idx = [int(x) for x in line[1:1 + n]]
                    for i in range(1, len(idx) - 1):
                        mesh.indices.extend([idx[0], idx[i], idx[i + 1]])

    def _read_binary(self, f, elements, mesh: Mesh, big_endian: bool) -> None:
        prefix = ">" if big_endian else "<"
        for el in elements:
            for _ in range(el["count"]):
                values = {}
                face_indices = None
                for prop in el["props"]:
                    if prop[0] == "scalar":
                        code, size = PLY_TYPES.get(prop[1], ("f", 4))
                        raw = f.read(size)
                        if len(raw) < size:
                            return
                        values[prop[2]] = struct.unpack(prefix + code, raw)[0]
                    else:
                        _, count_type, index_type, _name = prop
                        ccode, csize = PLY_TYPES.get(count_type, ("B", 1))
                        raw = f.read(csize)
                        if len(raw) < csize:
                            return
                        n = struct.unpack(prefix + ccode, raw)[0]
                        icode, isize = PLY_TYPES.get(index_type, ("i", 4))
                        raw = f.read(isize * n)
                        if len(raw) < isize * n:
                            return
                        face_indices = list(struct.unpack(prefix + icode * n, raw))
                if el["name"] == "vertex":
                    mesh.vertices.append((values.get("x", 0.0), values.get("y", 0.0),
                                          values.get("z", 0.0)))
                    if "nx" in values:
                        mesh.normals.append((values["nx"], values.get("ny", 0.0),
                                             values.get("nz", 0.0)))
                elif el["name"] == "face" and face_indices:
                    for i in range(1, len(face_indices) - 1):
                        mesh.indices.extend([face_indices[0], face_indices[i], face_indices[i + 1]])
