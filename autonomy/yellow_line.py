"""Linha amarela em plano calibrado; imagens RGB ou BGR, sem acesso a poses."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


class GroundProjection:
    """Homografia de pixels para (x frente, y esquerda), em metros no chao.

    A origem deve ser a projecao do centro do eixo dianteiro. A homografia
    pode vir dos parametros de uma camera virtual ou de pontos medidos no
    chao para uma camera real (apos corrigir a distorcao da lente).
    """

    def __init__(self, image_to_ground, resolution):
        self.matrix = np.asarray(image_to_ground, dtype=float)
        self.resolution = tuple(int(v) for v in resolution)
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all():
            raise ValueError("Homografia deve ser uma matriz 3x3 finita.")
        if abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError("Homografia singular.")
        if len(self.resolution) != 2 or min(self.resolution) <= 0:
            raise ValueError("Resolucao invalida.")
        self.inverse = np.linalg.inv(self.matrix)

    def ground_points(self, pixels):
        points = np.column_stack((np.asarray(pixels, dtype=float).reshape(-1, 2),
                                  np.ones(len(pixels)))) @ self.matrix.T
        valid = np.abs(points[:, 2]) > 1e-9
        xy = np.full((len(points), 2), np.nan)
        xy[valid] = points[valid, :2] / points[valid, 2, None]
        return xy

    def image_points(self, ground):
        points = np.column_stack((np.asarray(ground, dtype=float).reshape(-1, 2),
                                  np.ones(len(ground)))) @ self.inverse.T
        valid = np.abs(points[:, 2]) > 1e-9
        pixels = np.full((len(points), 2), np.nan)
        pixels[valid] = points[valid, :2] / points[valid, 2, None]
        return pixels


@dataclass(frozen=True)
class LineEstimate:
    cross_track: float  # m; positivo = linha a esquerda
    heading_error: float  # rad; tangente relativa a carro, positivo = esquerda
    curvature: float  # 1/m; maior modulo no trecho visivel
    confidence: float
    coefficients: tuple  # y(x) = a*x^2 + b*x + c
    min_x: float
    max_x: float
    residual: float  # RMS [m]
    circular: bool = False


class YellowLineDetector:
    def __init__(self, projection, color_order="RGB", min_x=0.35, max_x=3.0,
                 half_width=1.5, pixels_per_meter=100):
        if color_order not in ("RGB", "BGR"):
            raise ValueError("Ordem de cor deve ser RGB ou BGR.")
        self.projection = projection
        self.color_order = color_order
        self.min_x, self.max_x = min_x, max_x
        self.half_width = half_width
        self.ppm = pixels_per_meter
        self.size = (int(2 * half_width * self.ppm), int(max_x * self.ppm))
        self.ground_to_bird = np.array([
            [0, -self.ppm, half_width * self.ppm],
            [-self.ppm, 0, max_x * self.ppm],
            [0, 0, 1],
        ], dtype=float)
        self.mask = None
        self.bird_mask = None

    def detect(self, image):
        if image is None:
            return None
        width, height = self.projection.resolution
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError("Imagem deve ser uint8 HxWx3 na resolucao calibrada.")
        conversion = cv2.COLOR_RGB2HSV if self.color_order == "RGB" else cv2.COLOR_BGR2HSV
        hsv = cv2.cvtColor(np.ascontiguousarray(image), conversion)
        # OpenCV uint8: H em [0, 179]. Branco e asfalto sao rejeitados por S.
        self.mask = cv2.inRange(hsv, np.array([17, 90, 65], np.uint8),
                                np.array([38, 255, 255], np.uint8))
        self.bird_mask = cv2.warpPerspective(
            self.mask, self.ground_to_bird @ self.projection.matrix, self.size,
            flags=cv2.INTER_NEAREST,
        )
        # Fecha pequenos buracos sem apagar a faixa fina vista ao longe.
        self.bird_mask = cv2.morphologyEx(self.bird_mask, cv2.MORPH_CLOSE,
                                         np.ones((3, 3), np.uint8))
        points = []
        last_y = 0.0
        # Parte de perto: evita escolher outra linha na mesma imagem.
        for x in np.arange(self.min_x, self.max_x - 0.05, 0.05):
            row = int((self.max_x - x) * self.ppm)
            band = self.bird_mask[max(0, row - 2):min(self.size[1], row + 3)]
            columns = np.flatnonzero(np.count_nonzero(band, axis=0) >= 2)
            if not len(columns):
                continue
            runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
            candidates = [(self.half_width - float(np.mean(run)) / self.ppm, len(run))
                          for run in runs if 1 <= len(run) <= 0.30 * self.ppm]
            if not candidates:
                continue
            y, _ = min(candidates, key=lambda item: abs(item[0] - last_y))
            if points and abs(y - last_y) > 0.25:
                continue
            points.append((x, y))
            last_y = y

        if len(points) < 10:
            return None
        points = np.asarray(points)
        x, y = points.T
        if x[0] > 1.1 or np.ptp(x) < 0.45 - 1e-6:
            return None  # nao extrapola uma linha vista apenas no horizonte
        # O Stanley precisa da tangente junto ao eixo dianteiro. Ajustar os
        # 3 m inteiros mistura curvas opostas em um S e desloca essa tangente.
        local = x <= min(x[0] + 0.8, 1.4)
        x, y = x[local], y[local]
        if len(x) < 10 or np.ptp(x) < 0.45 - 1e-6:
            return None
        weights = 1 / (0.5 + x)
        coefficients = np.polyfit(x, y, 2, w=weights)
        # Rejeicao robusta de reflexos, falhas da mascara e objetos amarelos.
        errors = y - np.polyval(coefficients, x)
        mad = np.median(np.abs(errors - np.median(errors)))
        keep = np.abs(errors) < max(0.04, 3 * 1.4826 * mad)
        if np.count_nonzero(keep) < 10 or np.ptp(x[keep]) < 0.45 - 1e-6:
            return None
        x, y = x[keep], y[keep]
        coefficients = np.polyfit(x, y, 2, w=1 / (0.5 + x))
        residual = float(np.sqrt(np.mean((y - np.polyval(coefficients, x)) ** 2)))
        linear = np.polyfit(x, y, 1)
        linear_residual = float(np.sqrt(np.mean((y - np.polyval(linear, x)) ** 2)))
        if linear_residual < 0.005:
            # Evita inventar curvatura por quantizacao de uma reta em pixels.
            coefficients = np.array([0.0, *linear])
            residual = linear_residual
        if residual > 0.07:
            return None
        a, b, c = coefficients
        # Ponto da curva mais proximo do eixo dianteiro: x + y*y' = 0.
        roots = np.roots([2*a*a, 3*a*b, 1+b*b+2*a*c, b*c])
        candidates = [float(root.real) for root in roots
                      if abs(root.imag) < 1e-6 and -0.8 <= root.real <= 1.1]
        if not candidates:
            return None
        near_x = min(candidates, key=lambda xx: xx**2 + np.polyval(coefficients, xx)**2)
        near_y = float(np.polyval(coefficients, near_x))
        heading = math.atan(2 * a * near_x + b)
        cross_track = -near_x * math.sin(heading) + near_y * math.cos(heading)
        slopes = 2 * a * np.linspace(near_x, min(float(x[-1]), 2.0), 20) + b
        curvature = float(np.max(np.abs(2 * a / (1 + slopes**2) ** 1.5)))
        circular = False
        # Em arcos fechados a extrapolacao de uma parabola ate x=0 pode
        # inverter a tangente. Um arco circular local preserva a geometria.
        if curvature > 0.08:
            design = np.column_stack((2*x, 2*y, np.ones(len(x))))
            center_x, center_y, constant = np.linalg.lstsq(design, x*x+y*y, rcond=None)[0]
            radius_squared = constant + center_x**2 + center_y**2
            if radius_squared > 0:
                radius = math.sqrt(radius_squared)
                circle_residual = float(np.sqrt(np.mean(
                    (np.hypot(x-center_x, y-center_y) - radius)**2)))
                if (0.5 < radius < 12 and abs(center_y) > 0.3
                        and circle_residual < min(0.02, residual * 1.5)):
                    sign = math.copysign(1, center_y)
                    cross_track = sign * (math.hypot(center_x, center_y) - radius)
                    heading = math.atan2(-sign*center_x, abs(center_y))
                    curvature = 1 / radius
                    circular = True
        if abs(cross_track) > 1.0 or abs(heading) > math.radians(65):
            return None
        confidence = min(1.0, len(x) / 14) * max(0.0, 1 - residual / 0.07)
        return LineEstimate(float(cross_track), heading, curvature, confidence,
                            tuple(float(v) for v in coefficients),
                            float(x[0]), float(x[-1]), residual, circular)

    def annotate(self, image, estimate):
        if image is None:
            width, height = self.projection.resolution
            image = np.zeros((height, width, 3), np.uint8)
        output = np.ascontiguousarray(image).copy()
        if estimate is not None:
            x = np.linspace(estimate.min_x, estimate.max_x, 60)
            ground = np.column_stack((x, np.polyval(estimate.coefficients, x)))
            pixels = self.projection.image_points(ground)
            pixels = pixels[np.isfinite(pixels).all(axis=1)]
            if len(pixels) > 1:
                cv2.polylines(output, [pixels.astype(np.int32)], False, (0, 255, 0), 2)
            text = f"e={estimate.cross_track:+.3f}m psi={math.degrees(estimate.heading_error):+.1f}deg"
        else:
            text = "SEM LINHA - PARADA"
        cv2.putText(output, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return output
