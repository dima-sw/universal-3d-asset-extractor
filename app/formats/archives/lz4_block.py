"""Pure-python LZ4 block decompressor.

Game containers (Unity bundles among them) store raw LZ4 blocks, not frames.
Implemented here so the base install needs no native LZ4 dependency; if the
optional `lz4` package is present it is used instead because it is faster.
"""
from __future__ import annotations


class Lz4Error(Exception):
    pass


def decompress_block(src: bytes, uncompressed_size: int) -> bytes:
    """Decode one LZ4 block. Raises Lz4Error on malformed input."""
    try:
        import lz4.block as _native
        return _native.decompress(src, uncompressed_size=uncompressed_size)
    except ImportError:
        pass
    except Exception as exc:
        raise Lz4Error(str(exc))

    out = bytearray(uncompressed_size)
    src_len = len(src)
    i = 0
    o = 0
    while i < src_len:
        token = src[i]
        i += 1
        literal_len = token >> 4
        if literal_len == 15:
            while True:
                if i >= src_len:
                    raise Lz4Error("truncated literal length")
                b = src[i]
                i += 1
                literal_len += b
                if b != 255:
                    break
        if literal_len:
            end = i + literal_len
            if end > src_len or o + literal_len > uncompressed_size:
                raise Lz4Error("literal run overruns buffer")
            out[o:o + literal_len] = src[i:end]
            i = end
            o += literal_len
        if i >= src_len:
            break                                  # last sequence has no match
        if i + 2 > src_len:
            raise Lz4Error("truncated match offset")
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0 or offset > o:
            raise Lz4Error("invalid match offset %d" % offset)
        match_len = token & 0x0F
        if match_len == 15:
            while True:
                if i >= src_len:
                    raise Lz4Error("truncated match length")
                b = src[i]
                i += 1
                match_len += b
                if b != 255:
                    break
        match_len += 4
        if o + match_len > uncompressed_size:
            raise Lz4Error("match overruns buffer")
        start = o - offset
        for k in range(match_len):                 # byte-wise: overlaps are legal
            out[o + k] = out[start + k]
        o += match_len
    if o != uncompressed_size:
        raise Lz4Error("decoded %d bytes, expected %d" % (o, uncompressed_size))
    return bytes(out)


def decompress_lzma_alone(src: bytes, uncompressed_size: int) -> bytes:
    """Unity-style LZMA: 5 props bytes then the stream, no size field."""
    import lzma
    import struct
    if len(src) < 5:
        raise Lz4Error("truncated LZMA properties")
    header = src[:5] + struct.pack("<Q", uncompressed_size)
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
    return dec.decompress(header + src[5:], uncompressed_size)
