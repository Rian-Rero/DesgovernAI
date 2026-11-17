#!/usr/bin/env python3
"""Grade sugerida para calibrar a camera e marcadores para imprimir.

Com a calibracao atual (mesmo aproximada), preve onde cada ponto do chao cai na
imagem e sugere SO os pontos que a camera ve, bem espalhados (perto: a cada
25 cm; longe: ate +-1 m). Na camera de 11/2026 (virada um pouco para a
esquerda) so 20 dos 30 pontos da grade fixa apareciam, e quase nenhum perto.

Gera tambem um PDF para imprimir:
  pagina 1   mapa visto de cima com os pontos numerados e como medir;
  depois     um marcador por ponto: quadrado xadrez (o PONTO e onde os quadrados
             pretos se tocam), com o numero e as coordenadas. Branco e preto
             fosco aparecem no piso brilhante, onde fita e caneta somem no reflexo.

    python3 tools/grade_calibracao.py --pdf calibracao/marcadores_grade.pdf

A grade fica em calibracao/grade_sugerida.json; a janela de calibracao
(calibrar_camera_lateral.py) usa a mesma ordem e os mesmos numeros.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'calibracao' / 'camera_lateral.json'
GRID_FILE = ROOT / 'calibracao' / 'grade_sugerida.json'
CANDIDATE_X = (.6, .8, 1., 1.5, 2., 2.5, 3.)
CANDIDATE_Y = (1., .75, .5, .25, 0., -.25, -.5, -.75, -1.)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def predict_pixels(homography, ground):
    """Chao (m) -> pixel pela homografia pixel->chao (so pontos a frente da camera)."""
    ground = np.asarray(ground, float).reshape(-1, 2)
    q = np.c_[ground, np.ones(len(ground))] @ np.linalg.inv(np.asarray(homography, float)).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[:, :2]/q[:, 2:3]


def suggest_grid(homography, image_size=(640, 480), margin=15, per_row=5, xs=CANDIDATE_X, ys=CANDIDATE_Y):
    """Pontos (x, y) visiveis, na ordem de clique: de perto para longe, da esquerda
    para a direita na imagem (y decrescente). Ate per_row por distancia, os dois
    extremos visiveis e o resto espalhado entre eles."""
    width, height = image_size
    points = []
    for x in xs:
        row = [(float(x), float(y)) for y in sorted(ys, reverse=True)]
        pixels = predict_pixels(homography, row)
        inside = [point for point, (u, v) in zip(row, pixels)
                  if np.isfinite(u) and margin <= u < width - margin and margin <= v < height - margin]
        # Faixa larga visivel: a grade redonda (+-1, +-0,5, 0). Faixa estreita (perto):
        # os de 25 cm, espalhados, sempre com o centro se ele aparece.
        round_ = [point for point in inside if point[1] in (1., .5, 0., -.5, -1.)]
        if len(round_) >= 4:
            inside = round_[:per_row]
        elif len(inside) > per_row:
            keep = set(np.round(np.linspace(0, len(inside) - 1, per_row)).astype(int).tolist())
            center = [k for k, point in enumerate(inside) if point[1] == 0.]
            if center and center[0] not in keep:
                keep.discard(min(keep - {0, len(inside) - 1}, key=lambda k: abs(k - center[0])))
                keep.add(center[0])
            inside = [inside[k] for k in sorted(keep)]
        points.extend(inside)
    return points


def save_grid(points, path=GRID_FILE, source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'pontos_m': [list(p) for p in points], 'calibracao_usada': str(source or ''),
                                'ordem': 'perto -> longe; esquerda -> direita'}, indent=2) + '\n',
                    encoding='utf-8')
    return path


def load_grid(path=None):
    try:
        data = json.loads(Path(path or GRID_FILE).read_text(encoding='utf-8'))
        points = [(float(x), float(y)) for x, y in data['pontos_m']]
        return points or None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def side_text(y):
    return 'centro' if abs(y) < 1e-9 else f'{abs(y):.2f} m {"ESQ" if y > 0 else "DIR"}'.replace('.', ',')


def coordinates_text(x, y):
    return f'x = {x:.2f} m'.replace('.', ',') + f'   y = {side_text(y)}'


# ------------------------------------------------------------------ PDF
DPI = 150
A4 = (1240, 1754)  # 210 x 297 mm a 150 dpi
CM = DPI/2.54


def _font(size, bold=False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def _map_page(points, hidden):
    """Pagina 1: mapa visto de cima (frente para cima) e instrucoes."""
    from PIL import Image, ImageDraw
    page = Image.new('RGB', A4, 'white')
    draw = ImageDraw.Draw(page)
    draw.text((70, 50), 'Calibração da câmera: onde marcar', font=_font(46, True), fill='black')
    lines = [
        'Carro parado, rodas retas. Origem: CENTRO DO EIXO DIANTEIRO (x frente, y lados).',
        '1. Estique uma trena (ou barbante) do centro do eixo dianteiro reta para a frente, alinhada',
        '    com o carro. Marque nela as distâncias x da tabela.',
        '2. Em cada x, meça para os lados (y) na perpendicular: esquadro, ou triângulo 30/40/50 cm.',
        '    ESQ = esquerda do carro, olhando para a frente.',
        '3. Cole cada marcador com o CENTRO (onde os quadrados pretos se tocam) sobre o ponto.',
        '    Fita crepe nas bordas, papel liso e sem dobras. O número fica de frente para o carro.',
        '4. Piso brilhante: desligue ou desvie luzes que refletem para a câmera.',
        '5. GUI, aba de testes: "Foto p/ calibração"; clique no centro de cada marcador, na ordem',
        '    dos números (a janela mostra onde cada um deve aparecer e tem lupa).',
    ]
    for k, line in enumerate(lines):
        draw.text((70, 125 + 34*k), line, font=_font(24, k == 0), fill='black')
    # Mapa: y de +1,15 (esquerda, a esquerda do desenho) a -1,15; x de -0,35 a 3,15 (frente para cima).
    scale = 300.
    left, top = 620 - 1.15*scale, 470
    to_page = lambda x, y: (left + (1.15 - y)*scale, top + (3.15 - x)*scale)
    for x in np.arange(0., 3.01, .5):
        a, b = to_page(x, 1.15), to_page(x, -1.15)
        draw.line([a, b], fill='#dddddd', width=1)
        draw.text((a[0] - 70, a[1] - 14), f'{x:.1f} m'.replace('.', ','), font=_font(22), fill='#888888')
    for y in (1., .5, 0., -.5, -1.):
        a, b = to_page(-.1, y), to_page(3.15, y)
        draw.line([a, b], fill='#dddddd', width=1)
        draw.text((a[0] - 30, a[1] + 8), f'{y:+.1f}'.replace('.', ','), font=_font(20), fill='#888888')
    axis = [to_page(0., 0.), to_page(3.1, 0.)]
    draw.line(axis, fill='#3366cc', width=3)
    draw.text((axis[1][0] + 10, axis[1][1]), 'trena', font=_font(22), fill='#3366cc')
    # Carro: retangulo 0,26 m de largura, eixo dianteiro em x = 0.
    car = [to_page(.12, .13), to_page(-.36, -.13)]
    draw.rectangle(car, outline='black', width=3, fill='#eeeeee')
    a, b = to_page(0., .16), to_page(0., -.16)
    draw.line([a, b], fill='red', width=5)
    draw.text((b[0] + 10, b[1] - 14), 'eixo dianteiro (origem)', font=_font(22), fill='red')
    draw.text((to_page(-.25, .9)[0], to_page(-.25, .9)[1]), 'ESQ  (+y)', font=_font(24, True), fill='black')
    draw.text((to_page(-.25, -.55)[0], to_page(-.25, -.55)[1]), 'DIR  (-y)', font=_font(24, True), fill='black')
    for x, y in hidden:
        u, v = to_page(x, y)
        draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline='#bbbbbb', width=2)
    for number, (x, y) in enumerate(points, 1):
        u, v = to_page(x, y)
        draw.ellipse([u - 11, v - 11, u + 11, v + 11], fill='black')
        draw.text((u + 13, v - 15), str(number), font=_font(26, True), fill='black')
    note = (f'{len(points)} pontos numerados (a câmera vê). Círculos cinza: fora da imagem, não precisa marcar.'
            if hidden else f'{len(points)} pontos numerados.')
    draw.text((70, A4[1] - 80), note, font=_font(24), fill='black')
    return page


def _marker(number, x, y, size_cm, cell):
    """Um marcador: quadrado xadrez 2x2 (o ponto e o centro), numero e coordenadas."""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', cell, 'white')
    draw = ImageDraw.Draw(image)
    side = int(size_cm*CM)
    half = side//2
    cx, cy = cell[0]//2, 40 + half
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], outline='black', width=3)
    draw.rectangle([cx - half, cy - half, cx, cy], fill='black')
    draw.rectangle([cx, cy, cx + half, cy + half], fill='black')
    label_y = cy + half + 18
    draw.text((cx, label_y), str(number), font=_font(64, True), fill='black', anchor='mt')
    draw.text((cx, label_y + 78), coordinates_text(x, y), font=_font(30), fill='black', anchor='mt')
    draw.text((cx, label_y + 120), 'centro = onde os pretos se tocam', font=_font(20), fill='#666666',
              anchor='mt')
    # Linha de corte.
    draw.rectangle([0, 0, cell[0] - 1, cell[1] - 1], outline='#cccccc', width=1)
    return image


def markers_pdf(points, path, hidden=()):
    """PDF: mapa + marcadores (perto: 4 por pagina, 9 cm; x >= 2 m: 2 por pagina, 13 cm)."""
    from PIL import Image
    pages = [_map_page(points, hidden)]
    near = [(k, p) for k, p in enumerate(points, 1) if p[0] < 2.]
    far = [(k, p) for k, p in enumerate(points, 1) if p[0] >= 2.]
    layouts = ((near, (A4[0]//2, A4[1]//2), 9.), (far, (A4[0], A4[1]//2), 13.))
    for group, cell, size in layouts:
        per_page = (A4[0]//cell[0])*(A4[1]//cell[1])
        for start in range(0, len(group), per_page):
            page = Image.new('RGB', A4, 'white')
            for slot, (number, (x, y)) in enumerate(group[start:start + per_page]):
                column, row = slot % (A4[0]//cell[0]), slot//(A4[0]//cell[0])
                page.paste(_marker(number, x, y, size, cell), (column*cell[0], row*cell[1]))
            pages.append(page)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=DPI)
    return path, len(pages)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calibration', type=Path, default=CALIBRATION, help='calibracao atual (aproximada)')
    parser.add_argument('--pdf', type=Path, default=ROOT / 'calibracao' / 'marcadores_grade.pdf')
    parser.add_argument('--grade', type=Path, default=GRID_FILE, help='onde salvar a grade sugerida')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from fva_car.lateral import GroundCalibration
    calibration = GroundCalibration.from_file(args.calibration)
    points = suggest_grid(calibration.homography, calibration.image_size)
    every = [(float(x), float(y)) for x in CANDIDATE_X for y in (1., .5, 0., -.5, -1.)]
    hidden = [p for p in every if p not in points]
    save_grid(points, args.grade, args.calibration)
    path, pages = markers_pdf(points, args.pdf, hidden)
    print(f'{len(points)} pontos visiveis -> {args.grade}')
    print(f'{path}: {pages} paginas (imprima em tamanho real, sem "ajustar a pagina")')
    for number, (x, y) in enumerate(points, 1):
        print(f'{number:3d}  {coordinates_text(x, y)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
