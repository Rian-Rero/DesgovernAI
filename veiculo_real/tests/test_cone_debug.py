"""Gravador de debug dos cones: JSONL por quadro, eventos, limites e resumo."""
import json
import math
from pathlib import Path
import queue
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.cone_debug import ConeDebugRecorder, clean
from fva_car.cones import ConeObservation, RealConeDetector
from test_cones import scene


def observation(sequence, valid=True, **debug):
    frame = np.full((480, 640, 3), sequence % 255, np.uint8)
    return ConeObservation(
        timestamp=10. + .05*sequence, errors=(0., 0., 0.) if valid else None,
        reason='ok' if valid else 'sem cones', sequence=sequence, source_frame=frame,
        detection_seconds=.01, processing_seconds=.02, capture_seconds=.005,
        debug=dict({'valid': valid, 'trajectory': {'segment': 'reta'} if valid else {}}, **debug))


class ConeDebugRecorderTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def read(self):
        lines = (self.folder / 'cones_debug.jsonl').read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_one_record_per_new_frame_with_events_and_summary(self):
        recorder = ConeDebugRecorder(self.folder, pre_frames=2, post_frames=1)
        control = {'active': True, 'hold': False}
        for sequence in range(1, 6):
            recorder.record(observation(sequence), control)
            recorder.record(observation(sequence), control)  # mesmo quadro: ignorado
        recorder.record(observation(6, valid=False), {'active': True, 'hold': True, 'lost_s': math.nan})
        recorder.record(observation(7, valid=False), {'active': False})
        recorder.record(observation(8), {'active': True})
        recorder.close()
        records = self.read()
        self.assertEqual([r['frame'] for r in records], list(range(1, 9)))
        self.assertEqual(records[5]['events'], ['perda_cones'])
        self.assertIsNone(records[5]['control']['lost_s'])  # NaN vira null (JSON estrito)
        self.assertEqual(records[6]['events'], ['parada'])
        self.assertEqual(records[7]['events'], ['reaquisicao'])
        frames = sorted(p.name for p in (self.folder / 'cone_frames').glob('*.jpg'))
        # 2 antes + o evento + 1 depois, e o mesmo para os eventos seguintes.
        self.assertIn('f000006_perda_cones.jpg', frames)
        self.assertIn('f000004_perda_cones_antes.jpg', frames)
        self.assertIn('f000007_parada.jpg', frames)
        self.assertNotIn('f000001_perda_cones_antes.jpg', frames)
        summary = json.loads((self.folder / 'cones_summary.json').read_text())
        self.assertEqual(summary['frames'], 8)
        self.assertEqual(summary['valid'], 6)
        self.assertEqual(summary['longest_invalid_frames'], 2)
        self.assertEqual(summary['hold'], 1)
        self.assertEqual(summary['events']['perda_cones'], 1)
        self.assertEqual(summary['segments'], {'reta': 6})

    def test_side_flip_branch_and_jump_events(self):
        recorder = ConeDebugRecorder(self.folder)
        recorder.record(observation(1), {'active': True})
        recorder.record(observation(2, sides={'flipped': True}, branch=True), {'active': True, 'outlier': True})
        recorder.close()
        self.assertEqual(self.read()[1]['events'], ['bifurcacao', 'salto', 'troca_lado'])

    def test_event_frame_cap(self):
        recorder = ConeDebugRecorder(self.folder, max_event_frames=3, pre_frames=5)
        for sequence in range(1, 20):
            recorder.record(observation(sequence, valid=sequence % 2 == 0), {'active': True})
        recorder.close()
        self.assertEqual(len(list((self.folder / 'cone_frames').glob('*.jpg'))), 3)
        self.assertEqual(json.loads((self.folder / 'cones_summary.json').read_text())['event_frames_saved'], 3)

    def test_full_queue_drops_instead_of_blocking(self):
        recorder = ConeDebugRecorder(self.folder, queue_size=1)
        recorder.close()  # para a escrita; a fila passa a encher
        recorder.closed = False
        recorder.queue = queue.Queue(maxsize=1)
        for sequence in range(1, 5):
            recorder.record(observation(sequence), {'active': True})
        self.assertGreater(recorder.dropped, 0)

    def test_real_detector_debug_is_strict_json(self):
        calibration, frame = scene(bend=.1)
        recorder = ConeDebugRecorder(self.folder)
        detector = RealConeDetector(calibration)
        for sequence in range(1, 4):
            item = detector.observe_frame(frame, 10. + .05*sequence)
            from dataclasses import replace
            recorder.record(replace(item, sequence=sequence), {'active': True, 'steer_rad': np.float64(.1)})
        recorder.close()
        records = self.read()
        self.assertEqual(len(records), 3)
        self.assertIn(records[-1]['vision']['trajectory']['segment'], ('reta', 'curva_esq', 'curva_dir'))
        json.dumps(records, allow_nan=False)

    def test_clean_converts_numpy_and_nonfinite(self):
        self.assertEqual(clean({'a': np.int64(2), 'b': [np.float32(1.5), math.inf], 'c': np.bool_(True)}),
                         {'a': 2, 'b': [1.5, None], 'c': True})


if __name__ == '__main__':
    unittest.main()
