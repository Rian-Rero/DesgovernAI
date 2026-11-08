"""Classificador de candidatos a cone: rede pequena em ONNX, executada com cv2.dnn.

A cor (cone_path.ConePathDetector) gera candidatos com o ponto da base; aqui um
recorte quadrado alinhado pela base, com o lado proporcional a altura esperada
do cone naquela linha (homografia), vai a uma CNN de 3 classes:
nao_cone, cone_verde, cone_vermelho. So os candidatos confirmados seguem para
as bordas. O ponto da base continua vindo da geometria (preciso em pixels).

O mesmo extract_crop e usado para gerar o conjunto de treino
(extrair_recortes_cones.py) e no carro: o modelo ve exatamente o mesmo recorte.

Arquivos do modelo: <nome>.onnx + <nome>.json (classes, recorte, ajustes de cor
dos candidatos usados no treino e confianca minima). Roda no Raspberry so com o
OpenCV (cv2.dnn), sem PyTorch.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

CLASSES = ('nao_cone', 'cone_verde', 'cone_vermelho')
COLOR_CLASS = {'green': 'cone_verde', 'red': 'cone_vermelho'}
DEFAULT_CROP = {'size': 32, 'height_factor': 1.6, 'bottom_margin': .15, 'min_side': 12}
# Cores mais largas para gerar candidatos quando ha classificador: pegam o verde
# escuro/azulado na sombra (ensaio de 23/10) e o modelo descarta o ruido.
LOOSE_COLOR = {'green_h_max': 105, 'green_chroma_min': 5, 'green_angle_high': 255}


def crop_window(base, px_per_m, cone_height_m, crop=None):
    """(esquerda, topo, lado) em pixels do recorte de um candidato com base em `base`."""
    crop = {**DEFAULT_CROP, **(crop or {})}
    side = max(float(crop['min_side']), crop['height_factor']*cone_height_m*float(px_per_m))
    x, y = float(base[0]), float(base[1])
    bottom = y + crop['bottom_margin']*side
    return x - side/2, bottom - side, side


def extract_crop(frame, base, px_per_m, cone_height_m, crop=None):
    """Recorte BGR uint8 size x size; fora da imagem repete a borda."""
    crop = {**DEFAULT_CROP, **(crop or {})}
    left, top, side = crop_window(base, px_per_m, cone_height_m, crop)
    x0, y0 = int(round(left)), int(round(top))
    n = max(1, int(round(side)))
    height, width = frame.shape[:2]
    x1, y1 = x0+n, y0+n
    region = frame[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width)]
    if region.size == 0:
        region = np.zeros((1, 1, 3), np.uint8)
    pads = (max(0, -y0), max(0, y1-height), max(0, -x0), max(0, x1-width))
    if any(pads):
        region = cv2.copyMakeBorder(region, *pads, cv2.BORDER_REPLICATE)
    size = int(crop['size'])
    interpolation = cv2.INTER_AREA if n > size else cv2.INTER_LINEAR
    return cv2.resize(region, (size, size), interpolation=interpolation)


def softmax(logits):
    logits = np.asarray(logits, np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def to_blob(crops, size):
    """Entrada da rede: NCHW float32, BGR, 0-1 (igual no treino e no carro)."""
    return cv2.dnn.blobFromImages(list(crops), 1/255., (size, size), (0, 0, 0), swapRB=False, crop=False)


class ConeClassifier:
    def __init__(self, model_path=None, *, net=None, meta=None):
        """model_path: .onnx com o .json ao lado. net/meta: injecao (testes)."""
        if meta is None:
            if model_path is None:
                raise ValueError('Informe o caminho do modelo .onnx')
            meta_path = Path(model_path).with_suffix('.json')
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                raise FileNotFoundError(f'Metadados do classificador nao encontrados: {meta_path}') from None
        self.classes = tuple(meta.get('classes', CLASSES))
        missing = {'nao_cone', *COLOR_CLASS.values()} - set(self.classes)
        if missing:
            raise ValueError(f'Modelo sem as classes {sorted(missing)}')
        self.crop = {**DEFAULT_CROP, **meta.get('crop', {})}
        self.candidate_settings = dict(meta.get('candidate_settings', {}))
        self.min_confidence = float(meta.get('min_confidence', .6))
        if net is None:
            if not Path(model_path).is_file():
                raise FileNotFoundError(f'Modelo do classificador nao encontrado: {model_path}')
            net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net = net
        self.path = None if model_path is None else str(model_path)

    def predict(self, crops):
        """Probabilidades (N, classes) para recortes BGR size x size."""
        if not len(crops):
            return np.zeros((0, len(self.classes)))
        self.net.setInput(to_blob(crops, int(self.crop['size'])))
        return softmax(np.asarray(self.net.forward()).reshape(len(crops), -1))

    def classify(self, frame, bases, scale_rows, cone_height_m, min_confidence=None):
        """bases: {'green': Nx2, 'red': Mx2} em pixels de `frame`.

        Devolve {cor: (mascara de aceitos, confianca da classe da cor)}. Um
        candidato so fica se a classe mais provavel for a cone da mesma cor e
        com probabilidade >= min_confidence. Sem escala na linha da base (sem
        calibracao naquele ponto), o candidato e mantido como antes.
        """
        threshold = self.min_confidence if min_confidence is None else float(min_confidence)
        crops, owners = [], []
        output = {}
        last = len(scale_rows)-1
        for color, points in bases.items():
            points = np.asarray(points, float).reshape(-1, 2)
            keep = np.ones(len(points), bool)
            confidence = np.full(len(points), math.nan)
            for index, (x, y) in enumerate(points):
                px = float(scale_rows[min(max(int(round(y)), 0), last)])
                if math.isfinite(px) and px > 0:
                    crops.append(extract_crop(frame, (x, y), px, cone_height_m, self.crop))
                    owners.append((color, index))
            output[color] = (keep, confidence)
        probabilities = self.predict(crops)
        for (color, index), row in zip(owners, probabilities):
            keep, confidence = output[color]
            wanted = self.classes.index(COLOR_CLASS[color])
            confidence[index] = row[wanted]
            keep[index] = int(np.argmax(row)) == wanted and row[wanted] >= threshold
        return output


def load_classifier(value):
    """None, caminho do .onnx ou um classificador pronto (qualquer objeto com classify)."""
    if value is None or hasattr(value, 'classify'):
        return value
    return ConeClassifier(value)
