"""Candidatos a cone: cor em Lab, histerese, reflexo, ROI, tamanho e forma.

Cores dos distratores medidas no ensaio real de 23/10 (porta marrom/laranja,
placa, reflexos no piso brilhante), sem incluir a captura no repositorio.
"""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cone_path import ConePathDetector, ConeSettings, _apexes
from cone_tracking import transform_points
from test_cone_navigation import H, calibration, metric_scene


def hsv_bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def triangle(frame, x, base, height, half, color):
    cv2.fillConvexPoly(frame, np.array([(x, base-height), (x-half, base), (x+half, base)], np.int32), color)


def floor_pixel(x_m, y_m):
    u, v = transform_points([[x_m, y_m]], np.linalg.inv(H))[0]
    return int(round(u)), int(round(v))


class CandidateTests(unittest.TestCase):
    settings = ConeSettings(tracking=False)

    def detect(self, frame, settings=None, calibrated=True):
        detector = ConePathDetector(calibration() if calibrated else None)
        return detector.process(frame, settings or self.settings, timestamp=1.)

    def test_real_distractor_colors_are_rejected_by_lab_hue(self):
        frame = np.full((480, 640, 3), (160, 170, 175), np.uint8)
        # Porta laranja (H13 S95 V143), placa (H9 S46 V104), reflexo (H16 S48 V133):
        # vermelhos pelo HSV largo, fora da faixa de matiz Lab do cone.
        for x, color in ((150, hsv_bgr(13, 95, 143)), (300, hsv_bgr(9, 46, 104)), (450, hsv_bgr(16, 48, 133))):
            triangle(frame, x, 400, 60, 22, color)
        result = self.detect(frame, calibrated=False)
        self.assertEqual(len(result.red_points), 0)
        # Mesmo formato com a cor real dos cones (H176 S168 V105) e aceito.
        triangle(frame, 560, 400, 60, 22, hsv_bgr(176, 168, 105))
        self.assertEqual(len(self.detect(frame, calibrated=False).red_points), 1)
        # Desligando o filtro Lab, o HSV sozinho aceitava os distratores.
        loose = self.detect(frame, ConeSettings(tracking=False, lab_filter=False, red_hue=18), calibrated=False)
        self.assertGreaterEqual(len(loose.red_points), 3)

    def test_floor_reflection_does_not_move_the_base(self):
        frame = np.full((480, 640, 3), (160, 170, 175), np.uint8)
        triangle(frame, 320, 300, 80, 28, hsv_bgr(176, 168, 105))
        # Reflexo invertido, da mesma cor (piso muito brilhante), abaixo da base.
        cv2.fillConvexPoly(frame, np.array([(300, 300), (340, 300), (320, 330)], np.int32), hsv_bgr(176, 168, 105))
        result = self.detect(frame, calibrated=False)
        self.assertEqual(len(result.red_points), 1)
        self.assertLessEqual(abs(result.red_points[0][1] - 300), 3)

    def test_hysteresis_recovers_shaded_cone_without_new_blobs(self):
        frame = np.full((480, 640, 3), (160, 170, 175), np.uint8)
        # Cone com metade de cima iluminada e metade de baixo na sombra (S/V baixos).
        triangle(frame, 200, 360, 90, 32, hsv_bgr(176, 170, 140))
        shade = np.zeros(frame.shape[:2], np.uint8)
        shade[316:361] = 1
        mask = (shade == 1) & np.all(frame == hsv_bgr(176, 170, 140), axis=2)
        # Sombra: V=34 abaixo do minimo forte (35), croma Lab ~15 (ainda cone).
        frame[mask] = hsv_bgr(176, 255, 34)
        # Mancha isolada com a mesma cor fraca: sem pixel forte, nao entra.
        triangle(frame, 450, 360, 90, 32, hsv_bgr(176, 255, 34))
        with_h = self.detect(frame, calibrated=False)
        without = self.detect(frame, ConeSettings(tracking=False, hysteresis=False), calibrated=False)
        self.assertEqual(len(with_h.red_points), 1)
        self.assertLessEqual(abs(with_h.red_points[0][1] - 360), 3)
        # Sem histerese, a base fica na borda da luz (base errada) ou o cone some.
        self.assertTrue(len(without.red_points) == 0 or abs(without.red_points[0][1] - 360) > 10)

    def test_metric_size_rejects_shapes_far_from_cone_size_with_trusted_calibration(self):
        frame = np.full((480, 640, 3), 35, np.uint8)
        # A 1 m (100 px/m nesta homografia) o cone de 20 x 15 cm tem ~20 x 15 px.
        u, v = floor_pixel(1.0, .4)
        triangle(frame, u, v, 20, 7, (0, 220, 0))
        # "Camiseta" verde com formato de cone, mas 4x maior, no mesmo piso.
        u2, v2 = floor_pixel(1.0, -.5)
        triangle(frame, u2, v2, 80, 30, (0, 220, 0))
        result = self.detect(frame)
        self.assertTrue(result.debug['detection']['size_check'])
        bases = np.round(result.green_points).astype(int).tolist()
        self.assertEqual(len(bases), 1, bases)
        self.assertLessEqual(abs(bases[0][0]-u), 2)
        self.assertEqual(result.debug['detection']['green']['rejected'].get('tamanho'), 1)
        # Sem calibracao (sem escala), o formato sozinho aceita os dois.
        self.assertEqual(len(self.detect(frame, calibrated=False).green_points), 2)

    def test_far_small_cone_passes_area_scaled_by_distance(self):
        frame = np.full((480, 640, 3), 35, np.uint8)
        u, v = floor_pixel(2.8, .3)
        # Cone pequeno (area < min_area global de 28 px) mas do tamanho esperado a 2,8 m.
        triangle(frame, u, v, 6, 2, (0, 0, 240))  # cone esperado: ~7 x 5 px
        self.assertLess(cv2.countNonZero(cv2.inRange(frame, (0, 0, 200), (0, 0, 255))), 28)
        calibrated = self.detect(frame)
        plain = self.detect(frame, calibrated=False)
        self.assertEqual(len(calibrated.red_points), 1)
        self.assertEqual(len(plain.red_points), 0)

    def test_lab_lookup_table_matches_direct_computation(self):
        # A tabela (a*, b*) substitui cartToPolar por quadro: mesmo resultado.
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        settings = ConeSettings(green_chroma_min=6)
        green, red = ConePathDetector().segment(frame, settings, 200)
        region = cv2.medianBlur(frame[200:], 3)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        magnitude, angle = cv2.cartToPolar(lab[:, :, 1].astype(np.float32)-128,
                                           lab[:, :, 2].astype(np.float32)-128, angleInDegrees=True)
        green_c = (angle >= settings.green_angle_low) & (angle <= settings.green_angle_high) & (magnitude >= 6)
        red_c = ((angle >= settings.red_angle_low) | (angle <= settings.red_angle_high)) & (magnitude >= 12)
        strong_g = (cv2.inRange(hsv, (settings.green_h_min, settings.s_min, settings.v_min),
                                (settings.green_h_max, 255, 255)) > 0) & green_c
        strong_r = ((cv2.inRange(hsv, (0, settings.red_s_min, settings.red_v_min), (settings.red_hue, 255, 255))
                     | cv2.inRange(hsv, (179-settings.red_hue, settings.red_s_min, settings.red_v_min),
                                   (179, 255, 255))) > 0) & red_c
        # Pixels fortes sempre sobrevivem a histerese e ao fechamento.
        self.assertTrue(np.all(green[200:][strong_g]))
        self.assertTrue(np.all(red[200:][strong_r]))

    def test_cone_cut_by_image_side_keeps_its_base(self):
        frame = np.full((480, 640, 3), 35, np.uint8)
        u, v = floor_pixel(1.0, .4)
        # Cone de 20 x 15 cm a 1 m (~20 x 15 px) com metade esquerda fora da imagem.
        triangle(frame, 3, v, 20, 7, (0, 220, 0))
        triangle(frame, u, v, 20, 7, (0, 220, 0))
        result = self.detect(frame)
        bases = sorted(np.round(result.green_points).astype(int).tolist())
        self.assertEqual(len(bases), 2, bases)
        self.assertLessEqual(abs(bases[0][1]-v), 1)
        self.assertLessEqual(abs(bases[0][0]-3), 4)  # centro estimado pela meia base
        cut = self.detect(frame, ConeSettings(tracking=False, partial_cones=False))
        self.assertEqual(len(cut.green_points), 1)
        self.assertEqual(cut.debug['detection']['green']['rejected'].get('cortado_borda'), 1)

    def test_cone_glued_to_wall_noise_is_recovered(self):
        frame = np.full((480, 640, 3), (160, 170, 175), np.uint8)
        cone = hsv_bgr(176, 168, 105)
        triangle(frame, 320, 330, 70, 24, cone)
        # Linhas finas da mesma cor (batente da porta) tocando o topo do cone.
        cv2.line(frame, (300, 150), (318, 262), cone, 2)
        cv2.line(frame, (260, 200), (330, 262), cone, 2)
        result = self.detect(frame, calibrated=False)
        self.assertTrue(any(abs(x-320) <= 3 and abs(y-330) <= 3 for x, y in result.red_points),
                        result.red_points)

    def test_apex_count(self):
        def mask(*cones, shape=(140, 160)):
            image = np.zeros(shape, np.uint8)
            for x, base, height, half in cones:
                triangle(image, x, base, height, half, 1)
            return image.astype(bool)
        self.assertEqual(_apexes(mask((60, 120, 90, 30))), 1)
        self.assertEqual(_apexes(mask((50, 120, 90, 30), (100, 120, 90, 30))), 2)
        self.assertEqual(_apexes(mask((50, 130, 100, 35), (80, 90, 60, 20))), 2)


if __name__ == '__main__':
    unittest.main()
