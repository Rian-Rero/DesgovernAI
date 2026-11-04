"""Corredor de cones observado pela camera BGR calibrada do veiculo real.

Nao usa posicoes do simulador: projeta somente as bases visiveis no piso.
Convencao metrica: frente +x, esquerda +y, origem no eixo dianteiro.
"""
from dataclasses import dataclass, replace, asdict
import math
import time

import cv2
import numpy as np

from .lateral import GroundCalibration, LineObservation, fit_line_errors
from tools.cone_path import ConePathDetector, ConeSettings, render_cones


@dataclass(frozen=True)
class ConeObservation(LineObservation):
    reference_source: str = "cones"
    red_pixels: tuple = ()
    green_pixels: tuple = ()
    center_pixels: tuple = ()
    corridor_width_m: float = math.nan
    cone_result: object = None
    cone_settings: object = None



class RealConeDetector:
    """Adaptador metrico do MESMO detector utilizado pela interface HSV."""
    def __init__(self, calibration, settings=None, render_preview=False, processing_width=0):
        if not isinstance(calibration, GroundCalibration):
            raise ValueError('Cones requerem calibracao metrica da camera fixa.')
        self.calibration = calibration
        values = asdict(ConeSettings())
        changes = {} if settings is None else dict(settings)
        if set(changes)-set(values):
            raise ValueError('Ajuste de cones desconhecido: '+str(set(changes)-set(values)))
        values.update(changes)
        for name, value in values.items():
            default = getattr(ConeSettings(), name)
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            else:
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
            if not valid:
                raise ValueError('cone_settings.'+name+' tem valor invalido.')
        for name in ('roi_top', 'min_span', 'min_width', 'reference_y', 'camera_center', 'smoothing', 'min_confidence'):
            if values[name] > (99 if name == 'roi_top' else 100):
                raise ValueError('cone_settings.'+name+' fora do intervalo percentual.')
        if not 0 <= values['green_h_min'] <= values['green_h_max'] <= 179 or values['red_hue'] > 89:
            raise ValueError('Limites H dos cones invalidos.')
        if max(values['s_min'], values['v_min'], values['red_s_min'], values['red_v_min']) > 255:
            raise ValueError('S e V devem estar entre 0 e 255.')
        if not 0 < values['min_width_cm'] < values['max_width_cm'] or values['min_radius_cm'] <= 0:
            raise ValueError('Limites metricos dos cones invalidos.')
        self.settings = ConeSettings(**values)
        if (isinstance(processing_width, bool) or not isinstance(processing_width, int)
                or processing_width != 0 and not 160 <= processing_width <= calibration.image_size[0]):
            raise ValueError('vision_width deve ser 0 ou largura entre 160 e a resolucao calibrada.')
        w, h = calibration.image_size
        self.processing_size = (w, h) if not processing_width else (processing_width, round(h*processing_width/w))
        self.render_preview = render_preview
        # Mapas independentes: trocar a ordem das cores nao mistura o historico.
        self.detectors = {side: ConePathDetector(calibration) for side in (True, False)}
        self.green_left = self.settings.green_left

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
        small = frame if frame.shape[1::-1] == self.processing_size else cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        settings = replace(self.settings, green_left=self.green_left)
        result = self.detectors[self.green_left].process(small, settings, timestamp=timestamp)
        if not result.valid:
            settings = replace(self.settings, green_left=not self.green_left)
            candidate = self.detectors[settings.green_left].process(small, settings, timestamp=timestamp)
            if candidate.valid:
                result = candidate
                self.green_left = settings.green_left
            else:
                settings = replace(self.settings, green_left=self.green_left)
        metadata['detection_seconds'] = time.perf_counter()-started
        # O detector compartilha a homografia com correcao da resolucao reduzida.
        scale = np.asarray(self.calibration.image_size)/self.processing_size
        original = lambda pixels: tuple(map(tuple, (np.asarray(pixels)+.5)*scale-.5))
        metadata.update(red_pixels=original(result.red_points), green_pixels=original(result.green_points),
                        center_pixels=original(result.center), cone_result=result, cone_settings=settings)
        started = time.perf_counter()
        errors = None
        if result.valid and result.source == 'solo':
            try:
                # Stanley precisa dos erros junto ao eixo dianteiro, nao apenas
                # do deslocamento na primeira secao visivel usado na interface.
                errors = fit_line_errors(result.ground_center, [0., 0.], [1., 0.], with_curvature=True)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                pass
        metadata['geometry_seconds'] = time.perf_counter()-started
        valid = errors is not None and np.all(np.isfinite(errors))
        observation = ConeObservation(timestamp,
            errors=tuple(float(e) for e in errors) if valid else None,
            reason=result.status if valid or not result.valid else 'Cones sem cobertura metrica junto ao eixo dianteiro',
            corridor_width_m=result.corridor_width_m if valid else math.nan, **metadata)
        return self.with_preview(observation) if self.render_preview else observation

    def preview_bgr(self, observation):
        if observation is None or observation.source_frame is None:
            return None
        source = cv2.resize(observation.source_frame, self.processing_size)
        if observation.cone_result is None:
            return source
        return render_cones(source, observation.cone_result, observation.cone_settings)

    def with_preview(self, observation):
        if observation is None or observation.frame is not None:
            return observation
        image = self.preview_bgr(observation)
        return observation if image is None else replace(observation, frame=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
