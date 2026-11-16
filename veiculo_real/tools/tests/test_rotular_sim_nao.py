"""Rotulagem sim/nao (tools/rotular_sim_nao.py): preparar, rotular, exportar para o treino."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rotular_sim_nao as tool  # noqa: E402
from cone_classifier import CLASSES  # noqa: E402
from test_cone_candidates import hsv_bgr, triangle  # noqa: E402
from test_cone_navigation import H, metric_scene  # noqa: E402
from treinar_classificador_cones import load_datasets  # noqa: E402


class SimNaoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.calibration = base/'camera.json'
        self.calibration.write_text(json.dumps({'image_size': [640, 480], 'homography': H.tolist()}))
        run = base/'ensaios'/'20261110_093935'
        (run/'frames').mkdir(parents=True)
        for k in range(4):
            frame = metric_scene(lateral=.05*k)
            if k == 3:  # "mochila" vermelha escura no canto: candidato da cor larga
                triangle(frame, 600, 300, 30, 25, hsv_bgr(176, 168, 105))
            cv2.imwrite(str(run/'frames'/f'f{k:06d}.jpg'), frame)
        snapshot = base/'ensaios'/'20261023_034958'  # so a foto, sem frames/
        snapshot.mkdir()
        cv2.imwrite(str(snapshot/'vision_camera.jpg'), metric_scene(missing='green'))
        self.out = base/'rotulagem'

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, step=1):
        messages = []
        total = tool.prepare([Path(self.temp.name)/'ensaios'], self.out, self.calibration, step=step,
                             report=messages.append)
        return total, messages

    def test_prepare_finds_runs_and_photos_and_prelabels_with_the_detector(self):
        total, messages = self.prepare()
        items, labels = tool.load(self.out)
        self.assertEqual(total, len(items))
        self.assertEqual(labels, {})
        self.assertEqual({item['ensaio'] for item in items}, {'20261110_093935', '20261023_034958'})
        self.assertTrue(any(item['quadro'] == 'vision_camera.jpg' for item in items))
        self.assertGreater(sum(item['palpite'] for item in items), 10)  # os cones da cena
        self.assertEqual(np.load(self.out/tool.TRAIN_CROPS).shape, (total, 32, 32, 3))
        self.assertEqual(np.load(self.out/tool.VIEW_CROPS).shape, (total, 64, 64, 3))
        fewer = tool.prepare([Path(self.temp.name)/'ensaios'], self.out.with_name('passo2'), self.calibration,
                             step=2, report=lambda message: None)
        self.assertLess(fewer, total)

    def test_preparing_again_keeps_labels_and_adds_only_new_frames(self):
        total, _ = self.prepare()
        tool.save_labels(self.out, {0: True, 1: False})
        self.assertEqual(self.prepare()[0], 0)  # nada novo
        run = Path(self.temp.name)/'ensaios'/'20261110_093935'
        cv2.imwrite(str(run/'frames'/'f000010.jpg'), metric_scene())
        added, _ = self.prepare()
        items, labels = tool.load(self.out)
        self.assertGreater(added, 0)
        self.assertEqual(len(items), total + added)
        self.assertEqual([item['id'] for item in items], list(range(len(items))))
        self.assertEqual(labels, {0: True, 1: False})
        self.assertEqual(len(np.load(self.out/tool.TRAIN_CROPS)), len(items))

    def test_export_matches_the_training_format(self):
        self.prepare()
        items, _ = tool.load(self.out)
        labels = {item['id']: bool(item['palpite']) for item in items}
        wrong = next(item for item in items if item['palpite'])
        labels[wrong['id']] = False  # voce corrigiu um
        tool.save_labels(self.out, labels)
        dataset = self.out/'cones.npz'
        counts = tool.export(self.out, dataset)
        X, y, group, meta = load_datasets([dataset])
        self.assertEqual(len(y), len(items))
        self.assertEqual(X.shape[1:], (32, 32, 3))
        self.assertEqual(counts['nao_cone'], int(np.sum(y == CLASSES.index('nao_cone'))))
        self.assertGreater(counts['cone_verde'], 0)
        self.assertGreater(counts['cone_vermelho'], 0)
        self.assertEqual(len(np.unique(group)), len({(i['pasta'], i['quadro']) for i in items}))

    def test_only_labeled_items_are_exported(self):
        self.prepare()
        with self.assertRaises(ValueError):
            tool.export(self.out, self.out/'parcial.npz')
        tool.save_labels(self.out, {0: True, 1: False})
        tool.export(self.out, self.out/'parcial.npz')
        _, y, _, _ = load_datasets([self.out/'parcial.npz'])
        self.assertEqual(len(y), 2)

    @unittest.skipUnless(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'), 'sem tela')
    def test_grid_toggles_and_saves_each_page(self):
        self.prepare()
        labeler = tool.Labeler(self.out, columns=4, rows=2)
        try:
            first = labeler.page_items()[0]
            guess = labeler.current[first['id']]
            labeler.root.update()
            self.assertGreater(labeler.tile, 64)  # o recorte cresce com a janela
            x, y = labeler.tile_xy(0)
            labeler.toggle(type('E', (), {'x': x + 5, 'y': y + 5})())
            self.assertEqual(labeler.current[first['id']], not guess)
            labeler.next()
            _, saved = tool.load(self.out)
            self.assertEqual(saved[first['id']], not guess)
            self.assertEqual(len(saved), 8)
            self.assertEqual(labeler.page, 1)
            labeler.all_no()
            labeler.previous()  # volta: a pagina 2 tambem foi salva (toda "nao")
            _, saved = tool.load(self.out)
            self.assertEqual(len(saved), min(16, len(labeler.items)))
            self.assertEqual(labeler.current[first['id']], not guess)  # mostra o rotulo salvo
        finally:
            labeler.root.destroy()


if __name__ == '__main__':
    unittest.main()
