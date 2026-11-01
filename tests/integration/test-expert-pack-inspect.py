#!/usr/bin/env python3
"""Model-free subprocess checks for the expert-pack inspection boundary."""
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "engine/tools/expert-pack/create-moe-expert-pack.py"


class PackInspectionTests(unittest.TestCase):
    def inspect(self, *, offset=4096, size=8192, data_start=4096, count=1, nbytes=4096,
                expert=0, name=b"blk.0.ffn_up_exps.weight"):
        raw = struct.pack("<16sIIQQ", b"GGMLMOEPACKv1\0\0\0", 1, 40, count, data_start)
        raw += struct.pack("<128siIQQ", name, expert, 0, offset, nbytes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pack"
            path.write_bytes(raw + bytes(max(0, size - len(raw))))
            return subprocess.run([sys.executable, "-S", str(SCRIPT), "--inspect", "-o", str(path)],
                                  text=True, capture_output=True, timeout=10)

    def test_valid_payload(self):
        result = self.inspect()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("payload_bytes=4096", result.stdout)

    def test_invalid_ranges(self):
        for case in ({"size": 200}, {"offset": 0}, {"offset": 4095}, {"offset": 4097},
                     {"offset": 2**64 - 1}, {"nbytes": 2**64 - 1},
                     {"data_start": 40}, {"count": 2**64 - 1}, {"expert": -1},
                     {"name": b""}, {"name": b"n" * 128}, {"nbytes": 0}):
            with self.subTest(case=case):
                result = self.inspect(**case)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pack", result.stderr)
                self.assertNotIn("payload_bytes=", result.stdout)

    def test_valid_field_boundaries(self):
        result = self.inspect(expert=2**31 - 1, name=b"n" * 127, offset=8191, nbytes=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("payload_bytes=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
