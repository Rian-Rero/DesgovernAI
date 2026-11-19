#!/usr/bin/env python3
"""Rotulagem rapida "e cone? sim / nao" para treinar o classificador de cones.

Em vez de clicar na base de cada cone (rotular_cones.py), a ferramenta mostra os
CANDIDATOS que o carro julgaria (as mesmas cores largas do treino), ja recortados
como no carro, numa grade. Cada recorte vem pre-marcado pelo detector atual
(borda verde = cone, vermelha = nao). Voce so clica nos errados e passa a pagina.

1) Preparar (aceita ensaios e pastas de ensaios; rodar de novo so acrescenta os quadros novos):
    python3 tools/rotular_sim_nao.py preparar ~/fva_experimentos/carro/2026* --saida ~/fva_rotulagem
2) Clicar (para e continua quando quiser; salva a cada pagina):
    python3 tools/rotular_sim_nao.py rotular ~/fva_rotulagem
3) Exportar para o treino:
    python3 tools/rotular_sim_nao.py exportar ~/fva_rotulagem --out ~/fva_rotulagem/cones.npz
    python3 tools/treinar_classificador_cones.py ~/fva_rotulagem/cones.npz --out modelos/cones.onnx

Na grade:
    clique esquerdo     troca sim <-> nao
    clique direito      mostra o quadro inteiro com o candidato marcado
    Enter / espaco      confirma a pagina e vai para a proxima
    Backspace / b       volta uma pagina (para corrigir)
    t                   marca a pagina toda como "nao" (pagina de lixo)
    q / Esc             salva e sai

Quadros: <ensaio>/frames/f*.jpg (--record-frames) e, se nao houver, a foto
<ensaio>/vision_camera.jpg. --passo 2 usa um quadro sim, um nao: quadros vizinhos
repetem os mesmos cones e quase nao ensinam nada novo.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_classifier import CLASSES, CROP_CONE_HEIGHT_M, DEFAULT_CROP, LOOSE_COLOR, extract_crop  # noqa: E402
from cone_path import ConePathDetector, ConeSettings  # noqa: E402
from rotular_cones import DEFAULT_CALIBRATION, list_frames, load_calibration  # noqa: E402

QUEUE = 'fila.json'
LABELS = 'rotulos.json'
TRAIN_CROPS = 'recortes_treino.npy'
VIEW_CROPS = 'recortes_tela.npy'
VIEW_CROP = {'size': 64, 'height_factor': 2.6, 'bottom_margin': .3, 'min_side': 20}  # mais contexto na tela
MATCH_PX = 12.


def run_folders(paths, skip=None):
    """Ensaios (pastas com frames/ ou vision_camera.jpg) em qualquer nivel abaixo de `paths`."""
    import os
    skip = Path(skip).resolve() if skip else None
    found = []
    for path in map(Path, paths):
        for root, dirs, _ in os.walk(path):
            folder = Path(root)
            dirs[:] = sorted(d for d in dirs if d != 'frames' and (skip is None or (folder/d).resolve() != skip))
            if (list_frames(folder) or (folder/'vision_camera.jpg').exists()) and folder not in found:
                found.append(folder)
    return found


def frames_of(folder, step=1):
    frames = list_frames(folder)
    if frames:
        return frames[::max(1, int(step))]
    single = folder/'vision_camera.jpg'
    return [single] if single.exists() else []


def _near(point, points, radius=MATCH_PX):
    points = np.asarray(points, float).reshape(-1, 2)
    return bool(len(points)) and float(np.min(np.linalg.norm(points - point, axis=1))) <= radius


def prepare(paths, out, calibration_path=DEFAULT_CALIBRATION, step=2, max_per_frame=15, report=print):
    """Acrescenta a fila de `out` os candidatos dos quadros ainda nao vistos.

    Incremental: rodar de novo depois de coletar ensaios novos so processa os
    quadros novos; a fila e os rotulos ja feitos ficam. Devolve quantos entraram."""
    calibration = load_calibration(calibration_path)
    if calibration is None:
        raise ValueError('A preparacao precisa da calibracao da camera (escala do cone por linha).')
    loose = ConeSettings(tracking=False, **LOOSE_COLOR)
    strict = ConeSettings(tracking=False)
    detector = ConePathDetector(calibration)
    cone_height = CROP_CONE_HEIGHT_M  # recorte do classificador (cone_classifier.py)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    size, view_size = DEFAULT_CROP['size'], VIEW_CROP['size']
    if (out/QUEUE).exists():
        data = json.loads((out/QUEUE).read_text(encoding='utf-8'))
        queue, seen = data['itens'], set(data['meta'].get('quadros_vistos', []))
        train = list(np.load(out/TRAIN_CROPS))
        view = list(np.load(out/VIEW_CROPS))
    else:
        queue, seen, train, view = [], set(), [], []
    before = len(queue)
    rng = np.random.default_rng(len(seen))
    for folder in run_folders(paths, skip=out):
        count = 0
        for path in frames_of(folder, step):
            key = f'{folder.resolve()}/{path.name}'
            if key in seen:
                continue
            seen.add(key)
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[1::-1] != tuple(calibration.image_size):
                report(f'{path}: resolucao {frame.shape[1::-1]} difere da calibracao; pulado')
                continue
            found = detector.candidates(frame, loose)
            scale = found['scale']
            guess = detector.candidates(frame, strict)  # o que o detector atual aceita: pre-rotulo
            items = [(color, np.asarray(point, float)) for color in ('green', 'red') for point in found[color]]
            if len(items) > max_per_frame:
                items = [items[i] for i in sorted(rng.choice(len(items), max_per_frame, replace=False))]
            for color, point in items:
                px = float(scale[int(np.clip(round(point[1]), 0, len(scale) - 1))])
                if not np.isfinite(px) or px <= 0:
                    continue
                train.append(extract_crop(found['frame'], point, px, cone_height))
                view.append(extract_crop(frame, point, px, cone_height, VIEW_CROP))
                queue.append({'id': len(queue), 'ensaio': folder.name, 'pasta': str(folder.resolve()),
                              'quadro': path.name, 'cor': color, 'base': [round(float(v), 1) for v in point],
                              'palpite': _near(point, guess[color]),
                              # A camera muda de posicao entre dias: cada item guarda a sua.
                              'calibracao': str(Path(calibration_path).resolve())})
                count += 1
        if count:
            report(f'{folder.name}: {count} candidatos novos')
    np.save(out/TRAIN_CROPS, np.asarray(train, np.uint8).reshape(-1, size, size, 3))
    np.save(out/VIEW_CROPS, np.asarray(view, np.uint8).reshape(-1, view_size, view_size, 3))
    meta = {'calibracao': str(calibration_path), 'passo': step, 'max_por_quadro': max_per_frame,
            'candidatos': LOOSE_COLOR, 'recorte': DEFAULT_CROP, 'quadros_vistos': sorted(seen)}
    temporary = out/(QUEUE + '.tmp')
    temporary.write_text(json.dumps({'meta': meta, 'itens': queue}, ensure_ascii=False), encoding='utf-8')
    temporary.replace(out/QUEUE)
    labels = out/LABELS
    if not labels.exists():
        labels.write_text('{}', encoding='utf-8')
    return len(queue) - before


def load(out):
    out = Path(out)
    data = json.loads((out/QUEUE).read_text(encoding='utf-8'))
    labels = json.loads((out/LABELS).read_text(encoding='utf-8')) if (out/LABELS).exists() else {}
    return data['itens'], {int(k): bool(v) for k, v in labels.items()}


def save_labels(out, labels):
    path = Path(out)/LABELS
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({str(k): v for k, v in sorted(labels.items())}), encoding='utf-8')
    temporary.replace(path)


def export(out, target):
    """.npz no formato de extrair_recortes_cones.py (X, y, group, meta) so com o que foi rotulado."""
    out = Path(out)
    items, labels = load(out)
    crops = np.load(out/TRAIN_CROPS)
    chosen = [item for item in items if item['id'] in labels]
    if not chosen:
        raise ValueError('Nada rotulado ainda.')
    frames, runs = {}, {}
    X, y, group, run = [], [], [], []
    for item in chosen:
        is_cone = labels[item['id']]
        name = ('cone_verde' if item['cor'] == 'green' else 'cone_vermelho') if is_cone else 'nao_cone'
        X.append(crops[item['id']])
        y.append(CLASSES.index(name))
        group.append(frames.setdefault((item['pasta'], item['quadro']), len(frames)))  # validacao por quadro
        run.append(runs.setdefault(item['pasta'], len(runs)))  # validacao por ensaio
    y = np.asarray(y, np.int64)
    counts = {name: int(np.sum(y == i)) for i, name in enumerate(CLASSES)}
    meta = {'classes': list(CLASSES), 'crop': DEFAULT_CROP, 'candidate_settings': LOOSE_COLOR,
            'cone_height_m': CROP_CONE_HEIGHT_M,
            'match_px': MATCH_PX, 'sources': sorted({item['pasta'] for item in chosen}), 'counts': counts,
            'rotulagem': 'sim/nao (rotular_sim_nao.py)'}
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, X=np.asarray(X, np.uint8), y=y, group=np.asarray(group, np.int64),
                        run=np.asarray(run, np.int64), meta=json.dumps(meta))
    return counts


class Labeler:
    """Grade Tk de recortes; um clique troca sim/nao.

    Abre maximizada e os recortes crescem para ocupar a janela (F11 = tela cheia).
    Sem colunas/linhas, escolhe pela tela: recortes de ~1/8 da altura."""
    GAP = 10

    def __init__(self, out, columns=None, rows=None):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.out = Path(out)
        self.items, self.labels = load(self.out)
        self.view = np.load(self.out/VIEW_CROPS)
        self.root = tk.Tk()
        self.root.title('Rotulagem: e cone?')
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        # Tamanho de letra em pixels (negativo no Tk), pela altura da tela: legivel em 4K e em 1080p.
        self.font_px = max(12, round(screen_h/95))
        guess = max(96, screen_h//8)
        self.columns = columns or max(4, (screen_w - 40)//(guess + self.GAP))
        self.rows = rows or max(2, (screen_h - 10*self.font_px)//(guess + 2*self.font_px + self.GAP))
        self.per_page = self.columns*self.rows
        self.tile, self.origin, self.size = 64, (self.GAP, self.GAP), (0, 0)
        pending = [i for i, item in enumerate(self.items) if item['id'] not in self.labels]
        self.page = (pending[0] if pending else 0)//self.per_page
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, font=('TkDefaultFont', -round(1.3*self.font_px), 'bold'),
                 pady=4).pack()
        tk.Label(self.root, text='Pergunta: o objeto NA MARCA AMARELA (ponto = base) e um cone?   |   '
                                 'Clique = trocar   |   botao direito = quadro inteiro   |   Enter = proxima'
                                 '   |   Backspace = voltar   |   t = pagina toda "nao"   |   m = esconder marca'
                                 '   |   F11 = tela cheia',
                 fg='#555', font=('TkDefaultFont', -self.font_px)).pack()
        buttons = tk.Frame(self.root)
        buttons.pack(side='bottom', pady=(0, 8))
        font = ('TkDefaultFont', -self.font_px)
        tk.Button(buttons, text='< Voltar', font=font, command=self.previous).pack(side='left', padx=8)
        tk.Button(buttons, text='Pagina toda NAO (t)', font=font, command=self.all_no).pack(side='left', padx=8)
        tk.Button(buttons, text='Confirmar e proxima > (Enter)', font=font,
                  command=self.next).pack(side='left', padx=8)
        self.canvas = tk.Canvas(self.root, bg='#222', highlightthickness=0, width=screen_w*3//4,
                                height=screen_h*2//3)
        self.canvas.pack(fill='both', expand=True, padx=8, pady=8)
        self.canvas.bind('<Configure>', self.resized)
        self.canvas.bind('<Button-1>', self.toggle)
        self.canvas.bind('<Button-3>', self.show_frame)
        for key in ('<Return>', '<space>'):
            self.root.bind(key, lambda event: self.next())
        for key in ('<BackSpace>', 'b'):
            self.root.bind(key, lambda event: self.previous())
        self.root.bind('t', lambda event: self.all_no())
        self.show_mark = True
        self.root.bind('m', lambda event: self.toggle_mark())
        self.root.bind('<F11>', lambda event: self.fullscreen())
        for key in ('q', '<Escape>'):
            self.root.bind(key, lambda event: self.quit())
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        try:
            self.root.attributes('-zoomed', True)  # Linux (X11/XWayland)
        except tk.TclError:
            try:
                self.root.state('zoomed')  # Windows
            except tk.TclError:
                self.root.geometry(f'{screen_w}x{screen_h}+0+0')
        self.current = {}
        self.photos = []
        self._pending_draw = None
        self.load_page()

    def label_h(self):
        return round(1.6*self.font_px) + 6

    def layout(self, width, height):
        """Maior recorte quadrado que cabe colunas x linhas na area; a grade fica centrada."""
        gap, label = self.GAP, self.label_h()
        tile = min((width - gap)//self.columns - gap, (height - gap)//self.rows - gap - label)
        self.tile = max(32, int(tile))
        grid_w = self.columns*(self.tile + gap) + gap
        grid_h = self.rows*(self.tile + gap + label) + gap
        self.origin = (max(0, (width - grid_w)//2) + gap, max(0, (height - grid_h)//2) + gap)
        self.size = (width, height)

    def resized(self, event):
        if (event.width, event.height) == self.size:
            return
        self.layout(event.width, event.height)
        if self._pending_draw is not None:  # arrastando a borda: desenha so quando parar
            self.root.after_cancel(self._pending_draw)
        self._pending_draw = self.root.after(60, self.draw)

    def toggle_mark(self):
        self.show_mark = not self.show_mark
        self.draw()

    def mark(self, x, y):
        """Onde o candidato esta no recorte de tela: a base fica sempre no mesmo lugar
        (centro, 1 - bottom_margin) e um cone de verdade ocuparia ~1/height_factor de altura."""
        tile = self.tile
        base_x, base_y = x + tile/2, y + (1 - VIEW_CROP['bottom_margin'])*tile
        height = tile/VIEW_CROP['height_factor']
        half = .35*height
        width = max(2, tile//110)
        for color, extra in (('#000000', width + 2), ('#ffd400', width)):  # contorno preto: ve no claro e no escuro
            self.canvas.create_rectangle(base_x - half, base_y - height, base_x + half, base_y,
                                         outline=color, width=extra, dash=(6, 4))
        dot = max(4, tile//28)
        self.canvas.create_oval(base_x - dot, base_y - dot, base_x + dot, base_y + dot, fill='#ffd400',
                                outline='#000000', width=2)

    def fullscreen(self):
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))

    def page_items(self):
        start = self.page*self.per_page
        return self.items[start:start + self.per_page]

    def load_page(self):
        # Ja rotulado: o rotulo; senao, o palpite do detector.
        self.current = {item['id']: self.labels.get(item['id'], bool(item['palpite'])) for item in self.page_items()}
        self.draw()

    def tile_xy(self, index):
        row, column = divmod(index, self.columns)
        return (self.origin[0] + column*(self.tile + self.GAP),
                self.origin[1] + row*(self.tile + self.GAP + self.label_h()))

    def tile_at(self, x, y):
        step_x, step_y = self.tile + self.GAP, self.tile + self.GAP + self.label_h()
        column, row = int((x - self.origin[0])//step_x), int((y - self.origin[1])//step_y)
        index = row*self.columns + column
        items = self.page_items()
        if x >= self.origin[0] and y >= self.origin[1] and column < self.columns and row < self.rows \
                and index < len(items):
            return items[index]
        return None

    def draw(self):
        self._pending_draw = None
        self.canvas.delete('all')
        self.photos = []
        tile, line = self.tile, max(3, self.tile//40)
        for index, item in enumerate(self.page_items()):
            x, y = self.tile_xy(index)
            image = cv2.cvtColor(self.view[item['id']], cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_LINEAR)
            photo = self.ImageTk.PhotoImage(self.Image.fromarray(image))
            self.photos.append(photo)
            self.canvas.create_image(x, y, image=photo, anchor='nw')
            yes = self.current[item['id']]
            color = '#22dd44' if yes else '#ff3333'
            self.canvas.create_rectangle(x - line//2, y - line//2, x + tile + line//2, y + tile + line//2,
                                         outline=color, width=line)
            if self.show_mark:
                self.mark(x, y)
            self.canvas.create_text(x + tile/2, y + tile + self.label_h()/2 + 2,
                                    text='É CONE' if yes else 'NÃO É CONE', fill=color,
                                    font=('TkDefaultFont', -self.font_px, 'bold'))
        done = len(self.labels)
        total = len(self.items)
        pages = max(1, -(-total//self.per_page))
        self.status.set(f'Pagina {self.page + 1} de {pages}  |  rotulados {done} de {total}'
                        f' ({100*done/max(total, 1):.0f}%)  |  sim {sum(self.labels.values())}')

    def toggle(self, event):
        item = self.tile_at(event.x, event.y)
        if item is not None:
            self.current[item['id']] = not self.current[item['id']]
            self.draw()

    def show_frame(self, event):
        item = self.tile_at(event.x, event.y)
        if item is None:
            return
        folder = Path(item['pasta'])
        path = folder/'frames'/item['quadro']
        path = path if path.exists() else folder/item['quadro']
        frame = cv2.imread(str(path))
        if frame is None:
            return
        x, y = (int(round(v)) for v in item['base'])
        cv2.circle(frame, (x, y), 14, (0, 255, 255), 2)
        cv2.putText(frame, f'{item["ensaio"]} {item["quadro"]}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (0, 255, 255), 2)
        # Quadro inteiro no maior tamanho que cabe em ~80% da tela.
        scale = min(.8*self.root.winfo_screenwidth()/frame.shape[1],
                    .8*self.root.winfo_screenheight()/frame.shape[0])
        if scale > 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        window = self.tk.Toplevel(self.root)
        window.title('Quadro inteiro (clique ou Esc para voltar)')
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        label = self.tk.Label(window, image=photo)
        label.image = photo
        label.pack()
        window.bind('<Escape>', lambda e: window.destroy())
        window.bind('<Button-1>', lambda e: window.destroy())
        window.focus_set()

    def commit(self):
        self.labels.update(self.current)
        save_labels(self.out, self.labels)

    def next(self):
        self.commit()
        if (self.page + 1)*self.per_page < len(self.items):
            self.page += 1
            self.load_page()
        else:
            self.draw()
            self.status.set(self.status.get() + '  |  FIM: tudo rotulado. Exporte com "exportar".')

    def previous(self):
        self.commit()
        if self.page > 0:
            self.page -= 1
            self.load_page()

    def all_no(self):
        self.current = {key: False for key in self.current}
        self.draw()

    def quit(self):
        self.commit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='comando', required=True)
    p = commands.add_parser('preparar', help='acha e recorta os candidatos dos ensaios')
    p.add_argument('ensaios', nargs='+', type=Path)
    p.add_argument('--saida', type=Path, required=True, help='pasta da rotulagem (fora da pasta do codigo)')
    p.add_argument('--passo', type=int, default=2, help='usa 1 quadro a cada N')
    p.add_argument('--max-por-quadro', type=int, default=15)
    p.add_argument('--calibration', default=str(DEFAULT_CALIBRATION))
    r = commands.add_parser('rotular', help='abre a grade de sim/nao')
    r.add_argument('saida', type=Path)
    r.add_argument('--colunas', type=int, default=None, help='padrao: pelo tamanho da tela')
    r.add_argument('--linhas', type=int, default=None, help='padrao: pelo tamanho da tela')
    e = commands.add_parser('exportar', help='gera o .npz para treinar_classificador_cones.py')
    e.add_argument('saida', type=Path)
    e.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.comando == 'preparar':
        added = prepare(args.ensaios, args.saida, args.calibration, args.passo, args.max_por_quadro)
        items, labels = load(args.saida)
        print(f'{added} recortes novos; {len(items) - len(labels)} esperando rotulo em {args.saida}.'
              f' Agora: rotular_sim_nao.py rotular {args.saida}')
    elif args.comando == 'rotular':
        Labeler(args.saida, args.colunas, args.linhas).run()
    else:
        counts = export(args.saida, args.out)
        print(f'{args.out}: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
