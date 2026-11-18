#!/usr/bin/env python3
"""Calibra pixels -> (frente, esquerda) em metros, sem abrir motores ou servos.

Pelo notebook (o carro so tira a foto; a GUI faz isso no botao "Foto p/ calibracao"):

    python3 tools/calibrar_camera_lateral.py --image calibracao/fotos/foto.jpg

Por padrao, clique nos pontos medidos e digite suas coordenadas em metros
a partir do centro do eixo dianteiro (frente e esquerda).

Modo opcional --grade: marque no chao, com fita, x = 0,6 / 1 / 1,5 / 2 / 2,5 / 3 m
(frente, a partir do eixo DIANTEIRO) e y = +1 / +0,5 / 0 / -0,5 / -1 m (esquerda +).
A janela pede um ponto por vez, na ordem; clique nele ou "Pular" se nao aparece.
Com 5+ pontos mostra onde o proximo deve estar e o erro de cada um (deixando o
ponto de fora do ajuste): verde <= 3 cm, amarelo <= 6, vermelho acima (clique errado
ou medida errada: desfaca). Cones ficam a +-0,5-0,65 m: por isso a grade vai a +-1 m.

No carro, so a foto (sem janela):  python3 tools/calibrar_camera_lateral.py --capturar /tmp/foto.jpg

Grade sugerida (padrao quando existe): tools/grade_calibracao.py preve, pela
calibracao atual, quais pontos a camera ve e gera o PDF com o mapa e os
marcadores para imprimir (calibracao/grade_sugerida.json). Esta janela usa a
mesma ordem e numeracao, mostra desde o inicio onde cada ponto deve aparecer
(circulos numerados) e tem lupa no canto. --grade-fixa volta a grade de 30 pontos.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import GroundCalibration

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'calibracao' / 'camera_lateral.json'
GRID_X = (0.6, 1.0, 1.5, 2.0, 2.5, 3.0)
GRID_Y = (1.0, 0.5, 0.0, -0.5, -1.0)


def grid_points(xs=GRID_X, ys=GRID_Y):
    """Ordem de clique: de perto para longe, da esquerda para a direita na imagem."""
    return [(float(x), float(y)) for x in xs for y in ys]


def fit_homography(pixels, ground):
    matrix, _ = cv2.findHomography(np.asarray(pixels, np.float64), np.asarray(ground, np.float64), method=0)
    return matrix


def to_ground(matrix, pixels):
    points = cv2.perspectiveTransform(np.asarray(pixels, np.float64).reshape(-1, 1, 2), matrix)
    return points.reshape(-1, 2)


def reprojection_errors(pixels, ground):
    """Erro de cada ponto no chao, em metros: (no ajuste com todos, deixando-o de fora).

    O segundo e o que importa: um clique ou medida errada some no ajuste com todos
    (a homografia se curva para ele), mas aparece quando o ponto fica de fora."""
    pixels, ground = np.asarray(pixels, float), np.asarray(ground, float)
    fitted = np.full(len(pixels), np.nan)
    left_out = np.full(len(pixels), np.nan)
    if len(pixels) < 4:
        return fitted, left_out
    matrix = fit_homography(pixels, ground)
    if matrix is not None:
        fitted = np.linalg.norm(to_ground(matrix, pixels) - ground, axis=1)
    if len(pixels) >= 6:
        for index in range(len(pixels)):
            keep = np.arange(len(pixels)) != index
            matrix = fit_homography(pixels[keep], ground[keep])
            if matrix is not None:
                left_out[index] = np.linalg.norm(to_ground(matrix, pixels[index:index+1])[0] - ground[index])
    return fitted, left_out


def error_report(pixels, ground):
    fitted, left_out = reprojection_errors(pixels, ground)
    rows = []
    for number, ((x, y), a, b) in enumerate(zip(ground, fitted, left_out), 1):
        mark = '' if not np.isfinite(b) or b <= .03 else ('  <- confira' if b <= .06 else '  <- ERRADO?')
        rows.append(f'{number:2d}  x={x:4.2f} y={y:+5.2f}  ajuste {100*a:4.1f} cm  fora {100*b:5.1f} cm{mark}')
    return rows, fitted, left_out


def save_calibration(data, output):
    """Valida os pares medidos com a mesma geometria usada pela missao."""
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / 'candidate.json'
        candidate.write_text(json.dumps(data), encoding='utf-8')
        GroundCalibration.from_file(candidate)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return output


def capture_frame(index=None):
    """Foto na resolucao da missao (640x480). index None: a primeira camera que abrir."""
    for candidate in ([index] if index is not None else range(10)):
        camera = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        if camera.isOpened():
            index = candidate
            break
        camera.release()
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Camera {index} indisponivel (a missao esta rodando?).')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame = None
        for _ in range(5):
            ok, current = camera.read()
            if ok and current is not None:
                frame = current
        if frame is None:
            raise RuntimeError('Camera nao entregou uma imagem.')
        return frame
    finally:
        camera.release()


def select_points(frame, output):
    # Tkinter/Pillow tambem funcionam quando OpenCV foi instalado como headless.
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    from PIL import Image, ImageTk

    height, width = frame.shape[:2]
    scale = min(1., 1000 / width, 700 / height)
    display_size = (round(width * scale), round(height * scale))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    window = tk.Tk()
    window.title('Calibrar camera para o Stanley')
    tk.Label(window, text=(
        'Camera fixa, carro parado. Clique em pelo menos quatro pontos no chao,\n'
        'distribuidos em duas distancias e nos dois lados. Informe as medidas reais\n'
        'a partir do centro do eixo dianteiro: frente (+) e esquerda (+), em metros.'
    ), justify='left', padx=12, pady=12).pack()
    photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize(display_size))
    canvas = tk.Canvas(window, width=display_size[0], height=display_size[1])
    canvas.pack(padx=12)
    image_points, ground_points = [], []
    status = tk.StringVar(value='0 pontos medidos')
    tk.Label(window, textvariable=status, padx=12, pady=8).pack()

    def redraw():
        canvas.delete('all')
        canvas.create_image(0, 0, image=photo, anchor='nw')
        for number, (u, v) in enumerate(image_points, 1):
            x, y = u * display_size[0] / width, v * display_size[1] / height
            canvas.create_oval(x-5, y-5, x+5, y+5, fill='#ffb000', outline='black')
            canvas.create_text(x+12, y-10, text=str(number), fill='white')
        status.set(f'{len(image_points)} pontos medidos')

    def add_point(event):
        value = simpledialog.askstring(
            'Medida no chao',
            'Digite frente e esquerda, em METROS, separados por espaco.\n'
            'Exemplo de formato: 0.50 -0.20 (ponto a direita).', parent=window,
        )
        if value is None:
            return
        try:
            coordinates = [float(x.replace(',', '.')) for x in value.split()]
            if len(coordinates) != 2:
                raise ValueError('Informe dois numeros.')
        except ValueError as error:
            messagebox.showerror('Medida invalida', str(error), parent=window)
            return
        image_points.append([event.x * width / display_size[0],
                             event.y * height / display_size[1]])
        ground_points.append(coordinates)
        redraw()

    def undo():
        if image_points:
            image_points.pop()
            ground_points.pop()
            redraw()

    def save():
        data = {'image_size': [width, height], 'image_points_px': image_points,
                'ground_points_m': ground_points}
        try:
            saved = save_calibration(data, output)
        except (ValueError, OSError) as error:
            messagebox.showerror('Calibracao invalida', str(error), parent=window)
            return
        messagebox.showinfo('Calibracao salva', str(saved), parent=window)
        window.destroy()

    buttons = tk.Frame(window)
    buttons.pack(pady=12)
    tk.Button(buttons, text='Desfazer ultimo ponto', command=undo).pack(side='left', padx=6)
    tk.Button(buttons, text='Salvar calibracao', command=save).pack(side='left', padx=6)
    canvas.bind('<Button-1>', add_point)
    redraw()
    window.mainloop()


def select_grid(frame, output, targets, photo_path=None, hints=None):
    """Um ponto da grade por vez; mostra a previsao do proximo e o erro de cada um.

    hints: pixel previsto de cada alvo pela calibracao anterior (ou None): circulos
    numerados desde o inicio, para achar cada marcador na foto."""
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk

    height, width = frame.shape[:2]
    scale = min(1.6, 1100 / width, 760 / height)
    display_size = (round(width * scale), round(height * scale))
    window = tk.Tk()
    window.title('Calibrar camera (grade no chao)')
    photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(display_size))
    prompt = tk.StringVar()
    tk.Label(window, textvariable=prompt, font=('TkDefaultFont', 13, 'bold'), padx=12, pady=8).pack()
    views = tk.Frame(window)
    views.pack(padx=12)
    canvas = tk.Canvas(views, width=display_size[0], height=display_size[1], cursor='crosshair')
    canvas.pack(side='left')
    # Lupa: 5x em volta do mouse (piso brilhante: o centro do marcador e pequeno longe).
    ZOOM, RADIUS = 5, 16
    lens = tk.Canvas(views, width=2*RADIUS*ZOOM, height=2*RADIUS*ZOOM, bg='black')
    lens.pack(side='left', padx=(8, 0), anchor='n')
    lens_photo = {}
    table = tk.Text(window, height=8, width=70, font=('Courier', 10))
    state = {'index': 0, 'pixels': [], 'ground': [], 'skipped': []}

    def screen(u, v):
        return u * scale, v * scale

    def redraw():
        canvas.delete('all')
        canvas.create_image(0, 0, image=photo, anchor='nw')
        pixels, ground = state['pixels'], state['ground']
        rows, _, left_out = error_report(pixels, ground) if pixels else ([], [], [])
        for number, ((u, v), error) in enumerate(zip(pixels, left_out), 1):
            color = ('#ffb000' if not np.isfinite(error) else '#22cc44' if error <= .03
                     else '#ffdd00' if error <= .06 else '#ff3333')
            x, y = screen(u, v)
            canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline='black')
            canvas.create_text(x+10, y-10, text=str(number), fill='white', anchor='w')
        if hints is not None:
            for number, (u, v) in enumerate(hints, 1):
                if number - 1 < state['index'] or not (np.isfinite(u) and np.isfinite(v)):
                    continue
                x, y = screen(u, v)
                current = number - 1 == state['index']
                radius = 16 if current else 6
                canvas.create_oval(x-radius, y-radius, x+radius, y+radius, width=3 if current else 1,
                                   outline='magenta' if current else '#bbbbbb')
                canvas.create_text(x+radius+3, y+radius, text=str(number), anchor='w',
                                   fill='magenta' if current else '#dddddd')
        if state['index'] < len(targets):
            x_m, y_m = targets[state['index']]
            prompt.set(f'Ponto {state["index"]+1}/{len(targets)} (marcador {state["index"]+1}): clique no centro'
                       f' a {x_m:.2f} m a frente,'
                       f' {abs(y_m):.2f} m {"a esquerda" if y_m > 0 else "a direita" if y_m < 0 else "no centro"}'
                       + ('  (circulo magenta: onde deve estar)' if hints is not None else ''))
            if len(pixels) >= 5:
                matrix = fit_homography(ground, pixels)  # chao -> pixel: onde o proximo deve estar
                if matrix is not None:
                    u, v = to_ground(matrix, [[x_m, y_m]])[0]
                    x, y = screen(u, v)
                    canvas.create_line(x-12, y, x+12, y, fill='cyan', width=2)
                    canvas.create_line(x, y-12, x, y+12, fill='cyan', width=2)
        else:
            prompt.set('Grade completa: confira os erros e salve.')
        table.delete('1.0', 'end')
        table.insert('end', '\n'.join(rows[::-1]) or 'Clique nos pontos na ordem pedida (Pular se nao aparece).')

    def magnify(event):
        u, v = event.x / scale, event.y / scale
        x0, y0 = int(round(u)) - RADIUS, int(round(v)) - RADIUS
        patch = np.zeros((2*RADIUS, 2*RADIUS, 3), np.uint8)
        a, b = max(0, x0), max(0, y0)
        c, d = min(width, x0 + 2*RADIUS), min(height, y0 + 2*RADIUS)
        if c > a and d > b:
            patch[b-y0:d-y0, a-x0:c-x0] = frame[b:d, a:c]
        big = cv2.resize(patch, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        lens_photo['image'] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
        lens.delete('all')
        lens.create_image(0, 0, image=lens_photo['image'], anchor='nw')
        middle = RADIUS*ZOOM
        lens.create_line(middle-14, middle, middle+14, middle, fill='cyan')
        lens.create_line(middle, middle-14, middle, middle+14, fill='cyan')

    def add_point(event):
        if state['index'] >= len(targets):
            return
        state['pixels'].append([event.x / scale, event.y / scale])
        state['ground'].append(list(targets[state['index']]))
        state['index'] += 1
        redraw()

    def skip():
        if state['index'] < len(targets):
            state['skipped'].append(state['index'])
            state['index'] += 1
            redraw()

    def undo():
        if state['skipped'] and state['skipped'][-1] == state['index'] - 1:
            state['skipped'].pop()
        elif state['pixels']:
            state['pixels'].pop()
            state['ground'].pop()
        else:
            return
        state['index'] -= 1
        redraw()

    def save():
        _, left_out = reprojection_errors(state['pixels'], state['ground'])
        far = [g for g in state['ground'] if g[0] >= 2.]
        wide = [g for g in state['ground'] if abs(g[1]) >= .5]
        warnings = []
        if len(state['pixels']) < 8:
            warnings.append(f'so {len(state["pixels"])} pontos (ideal: 15+)')
        if not far or not wide:
            warnings.append('faltam pontos longe (x >= 2 m) ou nas laterais (|y| >= 0,5 m)')
        bad = int(np.sum(left_out > .06))
        if bad:
            warnings.append(f'{bad} ponto(s) com erro > 6 cm (vermelhos)')
        if warnings and not messagebox.askyesno('Salvar assim?', '\n'.join(warnings) + '\n\nSalvar mesmo assim?',
                                                parent=window):
            return
        data = {'image_size': [width, height], 'image_points_px': state['pixels'],
                'ground_points_m': state['ground'],
                'origem': 'centro do eixo dianteiro; x frente, y esquerda (m)',
                'erro_fora_cm': [None if not np.isfinite(e) else round(100*float(e), 1) for e in left_out]}
        if photo_path:
            data['foto'] = str(photo_path)
        try:
            saved = save_calibration(data, output)
        except (ValueError, OSError) as error:
            messagebox.showerror('Calibracao invalida', str(error), parent=window)
            return
        median = np.nanmedian(left_out) if np.isfinite(left_out).any() else float('nan')
        messagebox.showinfo('Calibracao salva', f'{saved}\nerro mediano (ponto fora do ajuste): {100*median:.1f} cm\n'
                            'Envie ao carro pela GUI (Enviar arquivos).', parent=window)
        window.destroy()

    buttons = tk.Frame(window)
    buttons.pack(pady=8)
    tk.Button(buttons, text='Pular ponto', command=skip).pack(side='left', padx=6)
    tk.Button(buttons, text='Desfazer', command=undo).pack(side='left', padx=6)
    tk.Button(buttons, text='Salvar calibracao', command=save).pack(side='left', padx=6)
    table.pack(padx=12, pady=(0, 12))
    canvas.bind('<Button-1>', add_point)
    canvas.bind('<Motion>', magnify)
    window.bind('<BackSpace>', lambda event: undo())
    window.bind('<space>', lambda event: skip())
    redraw()
    window.mainloop()


def grid_targets(args, image_size):
    """(alvos, pixels previstos ou None). Padrao: a grade sugerida (a mesma do PDF
    de marcadores; sem o arquivo, sugerida agora pela calibracao atual)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grade_calibracao import load_grid, predict_pixels, suggest_grid
    previous = None
    try:
        previous = GroundCalibration.from_file(args.output)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if args.grade_fixa or args.grade_x or args.grade_y:
        targets = grid_points(args.grade_x or GRID_X, args.grade_y or GRID_Y)
    else:
        targets = load_grid()
        if targets is None and previous is not None:
            targets = suggest_grid(previous.homography, tuple(image_size)) or None
        targets = targets or grid_points()
    if previous is None or tuple(previous.image_size) != tuple(image_size):
        return targets, None
    return targets, [tuple(p) for p in predict_pixels(previous.homography, targets)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--camera', type=int, help='Indice da camera USB para capturar uma foto')
    source.add_argument('--image', type=Path, help='Foto da camera fixa do carrinho')
    source.add_argument('--points', type=Path, help='JSON com pares ja medidos, sem interface grafica')
    source.add_argument('--capturar', type=Path, metavar='FOTO.jpg',
                        help='so tira a foto (no carro, sem janela) e sai')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--livre', action='store_true', help='clicar e digitar cada medida (padrao)')
    mode.add_argument('--grade', action='store_true', help='usar a grade sugerida')
    parser.add_argument('--grade-x', type=float, nargs='+', help='distancias a frente (m); com --grade-y: grade fixa')
    parser.add_argument('--grade-y', type=float, nargs='+', help='laterais, esquerda + (m)')
    parser.add_argument('--grade-fixa', action='store_true',
                        help='grade fixa de 30 pontos em vez da sugerida (calibracao/grade_sugerida.json)')
    args = parser.parse_args()
    try:
        if args.capturar:
            frame = capture_frame()
            args.capturar.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.capturar), frame):
                raise OSError(f'Nao foi possivel gravar {args.capturar}')
            print(f'Foto: {args.capturar} ({frame.shape[1]}x{frame.shape[0]})')
        elif args.points:
            data = json.loads(args.points.read_text(encoding='utf-8'))
            saved = save_calibration(data, args.output)
            print(f'Calibracao salva: {saved}')
            print('\n'.join(error_report(data['image_points_px'], data['ground_points_m'])[0]))
        else:
            frame = capture_frame(args.camera) if args.camera is not None else cv2.imread(str(args.image))
            if frame is None:
                raise ValueError('Nao foi possivel ler a imagem.')
            if not args.livre and (args.grade or args.grade_fixa or args.grade_x or args.grade_y):
                targets, hints = grid_targets(args, frame.shape[1::-1])
                select_grid(frame, args.output, targets, args.image, hints)
            else:
                select_points(frame, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
