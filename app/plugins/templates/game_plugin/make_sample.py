"""Writes sample files in the template's own format, so the plugin can be tried
end to end before any reverse engineering happens.

    python make_sample.py
    extractor scan .
"""
from __future__ import annotations

import struct
from pathlib import Path

MODEL_MAGIC = b"MYM0"
ARCHIVE_MAGIC = b"MYG0"


def build_model() -> bytes:
    vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    indices = [0, 1, 2, 0, 2, 3]
    out = bytearray(MODEL_MAGIC)
    out += struct.pack("<II", len(vertices), len(indices))
    for x, y, z in vertices:
        out += struct.pack("<3f", float(x), float(y), float(z))
    for i in indices:
        out += struct.pack("<H", i)
    return bytes(out)


def build_archive(payloads: list) -> bytes:
    header = bytearray(ARCHIVE_MAGIC + struct.pack("<I", len(payloads)))
    table_size = len(payloads) * 8
    offset = len(header) + table_size
    table = bytearray()
    blob = bytearray()
    for payload in payloads:
        table += struct.pack("<II", offset, len(payload))
        blob += payload
        offset += len(payload)
    return bytes(header + table + blob)


def main() -> None:
    here = Path(__file__).parent
    model = build_model()
    (here / "sample_model.bin").write_bytes(model)
    (here / "sample_archive.bin").write_bytes(build_archive([model, model]))
    print("wrote sample_model.bin and sample_archive.bin")


if __name__ == "__main__":
    main()
