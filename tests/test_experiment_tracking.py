# -*- coding: utf-8 -*-

import csv
import os
import tempfile
import unittest

from experiment_tracking import FrameStatus, FrameStatusRecorder, file_fingerprint


class TestFrameStatusRecorder(unittest.TestCase):
    def test_accounts_every_terminal_status(self):
        recorder = FrameStatusRecorder()
        recorder.record(FrameStatus(0, "a.jpg", "accepted", visual_fix_valid=True))
        recorder.record(FrameStatus(1, "b.jpg", "rejected_hold", reason="ambiguity"))
        recorder.record(FrameStatus(2, "c.jpg", "skipped", reason="missing_exif"))
        summary = recorder.summary(attempted_count=3)
        self.assertEqual(summary["accounted"], 3)
        self.assertEqual(summary["missing"], 0)
        self.assertAlmostEqual(summary["coverage"], 1.0 / 3.0)

    def test_duplicate_frame_is_rejected(self):
        recorder = FrameStatusRecorder()
        recorder.record(FrameStatus(0, "a.jpg", "accepted"))
        with self.assertRaises(ValueError):
            recorder.record(FrameStatus(0, "a.jpg", "failed"))

    def test_reports_conditional_error_and_dropout_recovery(self):
        recorder = FrameStatusRecorder()
        recorder.record(FrameStatus(0, "a.jpg", "accepted", error_m=10.0, latency_ms=5.0))
        recorder.record(FrameStatus(1, "b.jpg", "rejected_hold", error_m=80.0))
        recorder.record(FrameStatus(2, "c.jpg", "rejected_hold", error_m=90.0))
        recorder.record(FrameStatus(3, "d.jpg", "accepted", error_m=20.0, latency_ms=7.0))
        summary = recorder.summary(attempted_count=4)
        self.assertEqual(summary["accepted_error_m"]["p50"], 15.0)
        self.assertEqual(summary["accepted_error_m"]["max"], 20.0)
        self.assertEqual(summary["all_output_error_m"]["max"], 90.0)
        self.assertEqual(summary["dropout"]["max_consecutive_rejected_hold"], 2)
        self.assertEqual(summary["dropout"]["recovery_event_count"], 1)
        self.assertEqual(summary["dropout"]["recovery_frames_max"], 2)

    def test_csv_round_trip_has_one_row_per_frame(self):
        recorder = FrameStatusRecorder()
        recorder.record(FrameStatus(0, "a.jpg", "accepted", confidence=0.8))
        recorder.record(FrameStatus(1, "b.jpg", "failed", reason="io"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frames.csv")
            recorder.write_csv(path)
            with open(path, newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row["status"] for row in rows], ["accepted", "failed"])


class TestFingerprint(unittest.TestCase):
    def test_metadata_fingerprint(self):
        with tempfile.NamedTemporaryFile(delete=False) as stream:
            stream.write(b"abc")
            path = stream.name
        try:
            result = file_fingerprint(path, hash_mode="metadata")
            self.assertEqual(result["size_bytes"], 3)
            self.assertNotIn("sha256", result)
            hashed = file_fingerprint(path, hash_mode="sha256")
            self.assertEqual(
                hashed["sha256"],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
