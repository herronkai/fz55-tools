#!/usr/bin/env python3
"""Inspect an FZ55 update offline; never communicate with a camera."""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def parse_blocks(data):
    if data[:4] != b"WVDC":
        raise ValueError("Not a WVDC firmware container")
    blocks = []
    offset = 0x100
    while offset < len(data):
        if data[offset:offset + 8] != b"FUJITSU!":
            raise ValueError(f"Unexpected section header at {offset:#x}")
        kind, field, size = struct.unpack_from(">HHI", data, offset + 8)
        # Only directory/file records carry a path. Other records use this
        # field for a drive identifier or another section-specific value.
        name_size = field if kind in (3, 4) else 0
        start = offset + 16 + name_size
        end = start + size
        if end > len(data):
            raise ValueError(f"Section exceeds file bounds at {offset:#x}")
        name = data[offset + 16:start].decode("ascii")
        blocks.append(dict(offset=offset, type=kind, field=field,
                           payload_offset=start, size=size, name=name))
        offset = end
    return blocks


def inspect(path, output):
    data = path.read_bytes()
    blocks = parse_blocks(data)
    code_sections = [block for block in blocks if block["type"] == 2]
    if len(code_sections) != 1:
        raise ValueError("Expected one main code section")
    code = code_sections[0]
    origin = code["payload_offset"]
    code_end = origin + code["size"]
    base = 0xA0000000  # Inferred from DDR configuration and code literals.

    def word(address):
        offset = (address & ~1) - base + origin
        if not origin <= offset <= code_end - 4:
            raise ValueError("Pointer outside main section")
        return struct.unpack_from(">I", data, offset)[0]

    def string(address):
        offset = address - base + origin
        if not origin <= offset < code_end:
            raise ValueError("String pointer outside main section")
        end = data.find(b"\0", offset, min(offset + 512, code_end))
        if end < 0:
            raise ValueError("Unterminated string")
        return data[offset:end].decode("ascii")

    all_strings = [(match.start(), match.group().decode("ascii"))
                   for match in re.finditer(rb"[\x20-\x7e]{6,}", data)]
    raw_name = data.index(b"ss.raw\0", origin, code_end)
    raw_pointer = struct.pack(">I", base + raw_name - origin)
    command_refs = [match.start() for match in re.finditer(
        re.escape(raw_pointer), data[origin:code_end])]
    raw_records = []
    for relative in command_refs:
        offset = origin + relative
        name_ptr, handler, help_ptr = struct.unpack_from(">III", data, offset)
        try:
            raw_records.append(dict(name=string(name_ptr), handler=hex(handler),
                                    help=string(help_ptr), file_offset=hex(offset)))
        except (ValueError, UnicodeDecodeError):
            continue

    result = dict(
        file=str(path.resolve()), bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        project=data[32:48].split(b"\0")[0].decode("ascii"),
        version=data[48:64].split(b"\0")[0].decode("ascii"),
        code_base_inferred=hex(base), code_section=code,
        blocks=blocks, raw_monitor_records=raw_records,
    )
    # These table addresses were traced in v1.06, not assumed across releases.
    expected_sha = "5bcc3ccdc4d12e440c8aed9e12080ed78bb24b312696eb96a3a5f75535225747"
    if result["sha256"] == expected_sha:
        table = word(0xA0223E00)
        result["script_commands"] = [
            dict(name=string(word(table + i * 12)),
                 handler=hex(word(table + i * 12 + 4)))
            for i in range(20)
        ]

    output.mkdir(parents=True, exist_ok=True)
    (output / "firmware.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "strings.txt").write_text("".join(
        f"{offset:08x} {value}\n" for offset, value in all_strings))
    (output / "main-code.bin").write_bytes(data[origin:code_end])
    print(json.dumps({key: result[key] for key in
                      ("bytes", "sha256", "project", "version",
                       "raw_monitor_records")}, indent=2))
    print(f"Parsed {len(blocks)} complete sections; reports saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("firmware", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis/v1.06"))
    args = parser.parse_args()
    inspect(args.firmware, args.output)
