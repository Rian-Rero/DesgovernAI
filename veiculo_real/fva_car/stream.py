"""Transmissao MJPEG da visao da missao para um navegador na mesma rede.

Com o notebook no hotspot do carro, abra http://<ip-do-carro>:<porta>/.
Nao abre a camera: publica o que a thread de visao ja processou. So
desenha e comprime quadros enquanto houver alguem assistindo.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import cv2

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FVA - camera</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif;text-align:center}
img{width:100%;max-width:960px;image-rendering:pixelated}</style></head>
<body><img src="stream" alt="Aguardando quadros da missao..."></body></html>
"""


class VisionStream:
    def __init__(self, render, port=8080, fps=8., quality=70, host='0.0.0.0'):
        """render(observation, text) devolve uma imagem BGR ou None."""
        if not 0 < fps <= 30 or not 1 <= quality <= 100:
            raise ValueError('fps deve estar em (0, 30] e quality em [1, 100].')
        self.render = render
        self.period = 1. / fps
        self.quality = int(quality)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest = None
        self.jpeg = None
        self.sequence = 0
        self.viewers = 0
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # nao mistura acessos HTTP com a telemetria DATA

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self._send(200, 'text/html; charset=utf-8', PAGE)
                elif self.path == '/stream':
                    self._stream()
                else:
                    self._send(404, 'text/plain', b'Nao encontrado')

            def _send(self, code, content_type, body):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=quadro')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                with stream.condition:
                    stream.viewers += 1
                    stream.condition.notify_all()
                try:
                    sequence = 0
                    while not stream.stop_event.is_set():
                        sequence, frame = stream.wait_frame(sequence, timeout=1.)
                        if frame is None:
                            continue
                        self.wfile.write(b'--quadro\r\nContent-Type: image/jpeg\r\n'
                                         + f'Content-Length: {len(frame)}\r\n\r\n'.encode()
                                         + frame + b'\r\n')
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass  # navegador fechado
                finally:
                    with stream.condition:
                        stream.viewers -= 1

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.threads = [threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .2},
                                         daemon=True, name='camera-http'),
                        threading.Thread(target=self._encode_loop, daemon=True, name='camera-jpeg')]

    def start(self):
        for thread in self.threads:
            thread.start()
        return self

    def publish(self, observation, text=''):
        """Chamado no laco de controle: apenas guarda referencias."""
        self.latest = (observation, text)

    def wait_frame(self, after, timeout):
        """Devolve (sequencia, jpeg) mais novo que after, ou o atual no timeout."""
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > after or self.stop_event.is_set(),
                                    timeout)
            if self.sequence <= after:
                return after, None
            return self.sequence, self.jpeg

    def _encode_loop(self):
        last = None
        while not self.stop_event.wait(self.period):
            with self.condition:
                if not self.viewers:
                    self.condition.wait(.5)  # ninguem assistindo: nao gasta CPU
                    continue
            latest = self.latest
            if latest is None or latest is last:
                continue
            last = latest
            try:
                image = self.render(*latest)
                if image is None:
                    continue
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            except (cv2.error, ValueError, TypeError) as error:
                print(f'Transmissao da camera: {error}', flush=True)
                continue
            if ok:
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.sequence += 1
                    self.condition.notify_all()

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.threads[0].is_alive():
            self.server.shutdown()  # bloquearia se serve_forever nunca rodou
        self.server.server_close()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.)
