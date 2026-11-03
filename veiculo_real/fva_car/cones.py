"""Corredor de cones observado pela camera BGR calibrada do veiculo real.

Nao usa posicoes do simulador: projeta somente as bases visiveis no piso.
Convencao metrica: frente +x, esquerda +y, origem no eixo dianteiro.
"""
from dataclasses import dataclass, replace
import math
import time

import cv2
import numpy as np

from .lateral import GroundCalibration, LineObservation, fit_line_errors


@dataclass(frozen=True)
class ConeObservation(LineObservation):
    reference_source: str = "cones"
    red_pixels: tuple = ()
    green_pixels: tuple = ()
    center_pixels: tuple = ()
    corridor_width_m: float = math.nan


def corridor_center(left, right, min_width=.5, max_width=5.5):
    """Media de bordas avaliadas na MESMA distancia longitudinal.

    Nao associa cones apenas pela ordem da imagem (distancias desiguais).
    Usa somente a faixa longitudinal observada pelos dois lados.
    """
    boundaries = []
    for points in (left, right):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        points = points[np.all(np.isfinite(points), axis=1)]
        points = points[(points[:, 0] >= .05) & (points[:, 0] <= 5.) & (abs(points[:, 1]) <= 4.)]
        if len(points) < 2 or np.ptp(points[:, 0]) < .3:
            return None
        points = points[np.argsort(points[:, 0])]
        # Dois cones definem uma reta; tres ou mais permitem curvatura.
        degree = min(2, len(points)-1)
        design = np.vander(points[:, 0], degree+1)
        if np.linalg.cond(design) > 1e5:
            return None
        polynomial = np.polyfit(points[:, 0], points[:, 1], degree)
        if np.max(abs(np.polyval(polynomial, points[:, 0])-points[:, 1])) > .10:
            return None
        boundaries.append((points, polynomial))
    (lp, lc), (rp, rc) = boundaries
    start, end = max(lp[0, 0], rp[0, 0]), min(lp[-1, 0], rp[-1, 0])
    if start > 1.2 or end-start < .3:
        return None
    x = np.linspace(start, min(end, start+1.5), 60)
    yl, yr = np.polyval(lc, x), np.polyval(rc, x)
    slopes = (np.polyval(np.polyder(lc), x)+np.polyval(np.polyder(rc), x))/2
    widths = (yl-yr)/np.sqrt(1+slopes**2)
    if np.any(widths < min_width) or np.any(widths > max_width):
        return None
    # Bordas cruzando/divergindo muito sao deteccao ambigua, nao uma pista.
    if np.ptp(widths) > max(.25, .3*float(np.median(widths))):
        return None
    center = np.column_stack((x, (yl+yr)/2))
    errors = fit_line_errors(center, [0., 0.], [1., 0.], with_curvature=True)
    if errors is None or not np.all(np.isfinite(errors)):
        return None
    return center, tuple(float(e) for e in errors), float(np.median(widths))


class RealConeDetector:
    def __init__(self, calibration, settings=None, render_preview=False, processing_width=0):
        if not isinstance(calibration, GroundCalibration):
            raise ValueError('Cones requerem calibracao metrica da camera fixa.')
        self.calibration = calibration
        settings = {} if settings is None else dict(settings)
        defaults = {
            'red_hsv_ranges': [[[0, 100, 60], [20, 255, 255]], [[170, 100, 60], [179, 255, 255]]],
            'green_hsv_ranges': [[[35, 80, 50], [85, 255, 255]]],
            'min_area_px': 12., 'min_width_m': .5, 'max_width_m': 5.5,
        }
        if set(settings)-set(defaults):
            raise ValueError('Ajuste de cones desconhecido: '+str(set(settings)-set(defaults)))
        self.settings = dict(defaults, **settings)
        for name in ('red_hsv_ranges', 'green_hsv_ranges'):
            ranges = np.asarray(self.settings[name], dtype=float)
            if (ranges.ndim != 3 or ranges.shape[1:] != (2, 3) or not len(ranges)
                    or not np.all(np.isfinite(ranges)) or np.any(ranges < 0)
                    or np.any(ranges > [179, 255, 255]) or np.any(ranges[:, 0] > ranges[:, 1])):
                raise ValueError(name+' deve conter pares HSV OpenCV (H 0..179, S/V 0..255).')
            self.settings[name] = ranges.astype(np.uint8)
        for name in ('min_area_px', 'min_width_m', 'max_width_m'):
            value = self.settings[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' deve ser finito e positivo.')
        if self.settings['min_width_m'] >= self.settings['max_width_m']:
            raise ValueError('min_width_m deve ser menor que max_width_m.')
        if (isinstance(processing_width, bool) or not isinstance(processing_width, int)
                or processing_width != 0 and not 160 <= processing_width <= calibration.image_size[0]):
            raise ValueError('vision_width deve ser 0 ou largura entre 160 e a resolucao calibrada.')
        w, h = calibration.image_size
        self.processing_size = (w, h) if not processing_width else (processing_width, round(h*processing_width/w))
        self.render_preview = render_preview

    def _bases(self, hsv, ranges):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bases = []
        scale = np.asarray(self.calibration.image_size)/self.processing_size
        minimum = self.settings['min_area_px']/float(np.prod(scale))
        for contour in contours:
            if cv2.contourArea(contour) < minimum:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 3 or h/w < .5 or cv2.contourArea(contour) > .1*mask.size:
                continue
            if x == 0 or x+w >= hsv.shape[1] or y+h >= hsv.shape[0]:
                continue  # base fora da imagem nao define contato com o piso
            bottom = y+h-1
            xs = x+np.flatnonzero(mask[bottom, x:x+w])
            bases.append(((float(np.median(xs))+.5)*scale[0]-.5,
                          (bottom+.5)*scale[1]-.5))
        return np.asarray(bases, dtype=float).reshape(-1, 2)

    def observe_frame(self, frame, timestamp):
        metadata = dict(processing_size=self.processing_size)
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            return ConeObservation(timestamp, reason='Horario de captura invalido', **metadata)
        if frame is None:
            return ConeObservation(timestamp, reason='Camera sem imagem', **metadata)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return ConeObservation(timestamp, reason='Imagem deve ser BGR uint8', **metadata)
        metadata['source_frame'] = frame
        if frame.shape[1::-1] != self.calibration.image_size:
            return ConeObservation(timestamp, reason='Resolucao diferente da calibracao da camera', **metadata)
        started = time.perf_counter()
        small = frame if frame.shape[1::-1] == self.processing_size else cv2.resize(frame, self.processing_size)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        left = self._bases(hsv, self.settings['red_hsv_ranges'])
        right = self._bases(hsv, self.settings['green_hsv_ranges'])
        metadata.update(red_pixels=tuple(map(tuple, left)), green_pixels=tuple(map(tuple, right)),
                        detection_seconds=time.perf_counter()-started)
        started = time.perf_counter()
        try:
            red_ground = self.calibration.project_pixels(left)
            green_ground = self.calibration.project_pixels(right)
            result = corridor_center(red_ground, green_ground,
                                     self.settings['min_width_m'], self.settings['max_width_m'])
            if result is None:
                result = corridor_center(green_ground, red_ground,
                                         self.settings['min_width_m'], self.settings['max_width_m'])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            result = None
        metadata['geometry_seconds'] = time.perf_counter()-started
        if result is None:
            observation = ConeObservation(timestamp, reason='Cones sem duas bordas metricas confiaveis', **metadata)
        else:
            center, errors, width = result
            inverse = np.linalg.inv(self.calibration.homography)
            pixels = cv2.perspectiveTransform(center.reshape(-1, 1, 2), inverse).reshape(-1, 2)
            observation = ConeObservation(timestamp, errors=errors, reason='Centro dos cones valido',
                                          center_pixels=tuple(map(tuple, pixels)), corridor_width_m=width, **metadata)
        return self.with_preview(observation) if self.render_preview else observation

    def preview_bgr(self, observation):
        if observation is None or observation.source_frame is None:
            return None
        display = cv2.resize(observation.source_frame, self.processing_size).copy()
        scale = np.asarray(self.processing_size)/self.calibration.image_size
        for pixels, color in ((observation.red_pixels, (0, 0, 255)), (observation.green_pixels, (0, 255, 0))):
            for point in pixels:
                cv2.circle(display, tuple(np.rint((np.asarray(point)+.5)*scale-.5).astype(int)), 4, color, -1)
        if observation.center_pixels:
            points = np.rint((np.asarray(observation.center_pixels)+.5)*scale-.5)
            points = np.clip(points, -10000, 10000).astype(np.int32)
            cv2.polylines(display, [points], False, (255, 255, 0), 2)
        return display

    def with_preview(self, observation):
        if observation is None or observation.frame is not None:
            return observation
        image = self.preview_bgr(observation)
        return observation if image is None else replace(observation, frame=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class PriorityNavigationDetector:
    """Mesma imagem: cones primeiro, linha amarela como segunda referencia."""
    def __init__(self, cones, line):
        self.cones, self.line = cones, line
        self.processing_size = cones.processing_size

    def observe_frame(self, frame, timestamp):
        cones = self.cones.observe_frame(frame, timestamp)
        if cones.valid:
            return cones
        line = self.line.observe_frame(frame, timestamp)
        reason = ('Linha amarela (sem corredor de cones valido)' if line.valid else
                  f'Sem referencia: {cones.reason}; {line.reason}')
        return replace(line, reason=reason,
                       detection_seconds=cones.detection_seconds+line.detection_seconds,
                       geometry_seconds=cones.geometry_seconds+line.geometry_seconds)

    def _detector(self, observation):
        return self.cones if getattr(observation, 'reference_source', 'line') == 'cones' else self.line

    def with_preview(self, observation):
        return self._detector(observation).with_preview(observation)

    def preview_bgr(self, observation):
        return self._detector(observation).preview_bgr(observation)
