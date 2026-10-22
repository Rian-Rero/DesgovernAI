#!/usr/bin/env python3
"""Calibra pixels -> (frente, esquerda) em metros, sem abrir motores ou servos."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fva_car.lateral import GroundCalibration

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'calibracao' / 'camera_lateral.json'


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


def capture_frame(index):
    camera = cv2.VideoCapture(index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Camera {index} indisponivel.')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--camera', type=int, help='Indice da camera USB para capturar uma foto')
    source.add_argument('--image', type=Path, help='Foto da camera fixa do carrinho')
    source.add_argument('--points', type=Path, help='JSON com pares ja medidos, sem interface grafica')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.points:
            data = json.loads(args.points.read_text(encoding='utf-8'))
            print(f'Calibracao salva: {save_calibration(data, args.output)}')
        else:
            frame = capture_frame(args.camera) if args.camera is not None else cv2.imread(str(args.image))
            if frame is None:
                raise ValueError('Nao foi possivel ler a imagem.')
            select_points(frame, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
