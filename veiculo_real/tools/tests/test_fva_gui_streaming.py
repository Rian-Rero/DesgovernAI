"""Regressions for live telemetry. Run with unittest and requirements_gui.txt."""

import contextlib
import importlib.util
import os
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import paramiko
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


GUI_PATH = Path(__file__).resolve().parents[1] / "fva_gui_multiplataforma.py"
SPEC = importlib.util.spec_from_file_location("fva_gui", GUI_PATH)
gui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gui)


class LocalSSHServer(paramiko.ServerInterface):
    """Runs only the test's benign commands over a local socket pair."""

    def __init__(self):
        self.finished = threading.Event()
        self.workers = []
        self.errors = []

    def check_auth_password(self, username, password):
        return paramiko.AUTH_SUCCESSFUL

    def get_allowed_auths(self, username):
        return "password"

    def check_channel_request(self, kind, channel_id):
        return paramiko.OPEN_SUCCEEDED

    def check_channel_pty_request(self, *args):
        raise AssertionError("The live reader must not request a PTY")

    def check_channel_exec_request(self, channel, command):
        worker = threading.Thread(
            target=self.run_command, args=(channel, command), daemon=True,
        )
        self.workers.append(worker)
        worker.start()
        return True

    def run_command(self, channel, command):
        def forward(pipe, send):
            try:
                while True:
                    chunk = os.read(pipe.fileno(), 4096)
                    if not chunk:
                        break
                    send(chunk)
            except Exception as exc:
                self.errors.append(exc)
            finally:
                pipe.close()

        try:
            with subprocess.Popen(
                ["/bin/sh", "-c", command.decode()],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ) as process:
                streams = [
                    threading.Thread(target=forward, args=(process.stdout, channel.sendall)),
                    threading.Thread(target=forward, args=(process.stderr, channel.sendall_stderr)),
                ]
                for worker in streams:
                    worker.start()
                rc = process.wait(timeout=10)
                for worker in streams:
                    worker.join(timeout=3)
                channel.send_exit_status(rc)
                self.finished.set()
                channel.shutdown_write()
        except Exception as exc:
            self.errors.append(exc)
            self.finished.set()
        finally:
            channel.close()


@contextlib.contextmanager
def local_ssh_client():
    client_socket, server_socket = socket.socketpair()
    transport = paramiko.Transport(server_socket)
    transport.add_server_key(paramiko.RSAKey.generate(2048))
    server = LocalSSHServer()
    errors = []

    def start_server():
        try:
            transport.start_server(server=server)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=start_server, daemon=True)
    worker.start()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            "local-test", sock=client_socket, username="test", password="test",
            allow_agent=False, look_for_keys=False, timeout=5,
        )
        yield client, server
    finally:
        client.close()
        transport.close()
        worker.join(timeout=3)
        for command_worker in server.workers:
            command_worker.join(timeout=3)
        if errors or server.errors:
            raise AssertionError(errors + server.errors)


class PlotHarness:
    """Production receiver and plotter with a real Agg canvas, without Tk."""

    _execute_remote_command = gui.RsyncGUI._execute_remote_command
    _handle_remote_line = gui.RsyncGUI._handle_remote_line
    _record_stream_chunk = gui.RsyncGUI._record_stream_chunk
    _record_stream_progress = gui.RsyncGUI._record_stream_progress
    _store_telemetry_sample = gui.RsyncGUI._store_telemetry_sample
    _check_plot_health = gui.RsyncGUI._check_plot_health
    update_plot = gui.RsyncGUI.update_plot
    _plot_tick = gui.RsyncGUI._plot_tick
    _handle_plot_error = gui.RsyncGUI._handle_plot_error
    NO_TELEMETRY_WARNING_SECONDS = 3
    TELEMETRY_STALE_WARNING_SECONDS = 3
    PLOT_REFRESH_MS = 66

    def __init__(self):
        self.telemetry_lock = threading.Lock()
        self.telemetry = {}
        self._plot_stats = gui.RsyncGUI._new_plot_stats()
        self._active_stream = None
        self._plot_dirty = False
        self._plot_last_error = None
        self._plot_session_active = True
        self._plot_session_started_at = time.monotonic()
        self._plot_stale_warned = False
        self._plot_no_data_warned = False
        self.callbacks = queue.Queue()
        self.diagnostics = []
        self.terminal = []
        self.scheduled = []
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasAgg(self.fig)
        self.plot_var = type("Value", (), {"get": lambda _: "Velocidade"})()

    def ui(self, func, *args):
        self.callbacks.put((func, args))

    def drain_callbacks(self):
        while not self.callbacks.empty():
            func, args = self.callbacks.get_nowait()
            func(*args)

    def _plot_diag_write(self, level, message, event_time=None):
        self.diagnostics.append((level, message))

    def _set_plot_status(self, text, level="INFO"):
        self.status = text

    def cmdlog_write(self, message):
        self.terminal.append(message)

    def after(self, delay, callback):
        self.scheduled.append(callback)


class StreamingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FVA_TEST_TK") == "1", "Set FVA_TEST_TK=1 for real Tk testing")
    def test_tk_mainloop_renders_before_remote_exit(self):
        script = (
            "import time,sys; print('startup warning',file=sys.stderr); "
            "[(print(f'DATA,{i / 5},0,0,1,1,0,0,0,0'),time.sleep(.15)) "
            "for i in range(8)]; time.sleep(.5)"
        )
        with local_ssh_client() as (client, server), patch.object(gui.RsyncGUI, "refresh_ips"):
            app = gui.RsyncGUI()
            app.tab_cmds.master.select(app.tab_cmds)
            app.devices["vermelho"]["ip"] = "local-test"
            app.devices["vermelho"]["var"].set(1)
            app.dest_entry.delete(0, "end")
            app.dest_entry.insert(0, "/tmp")
            app.cmd_text.delete("1.0", "end")
            app.cmd_text.insert("end", f"python3 -c {shlex.quote(script)}")
            app._connect_ssh = lambda *args: client
            errors = []
            observed_live_frame = []
            started_at = time.monotonic()

            def observe():
                try:
                    if not observed_live_frame and app._plot_stats["live_frames"]:
                        self.assertFalse(server.finished.is_set(), "Tk rendered only after exit")
                        self.assertGreater(len(app.ax.lines[0].get_ydata()), 0)
                        pixels = np.asarray(app.canvas.buffer_rgba())
                        self.assertGreater(np.unique(pixels.reshape(-1, 4), axis=0).shape[0], 10)
                        observed_live_frame.append(time.monotonic() - started_at)
                    if not app._plot_session_active:
                        self.assertTrue(observed_live_frame, app.plot_diagnostics.get("1.0", "end"))
                        self.assertEqual(app._plot_stats["valid_samples"], 8)
                        logs = app.plot_diagnostics.get("1.0", "end")
                        self.assertIn("Marcador de início recebido", logs)
                        self.assertIn("Primeiro quadro em tempo real", logs)
                        self.assertEqual(str(app.run_cmds_button["state"]), "normal")
                        app.destroy()
                        return
                    self.assertLess(time.monotonic() - started_at, 7, "GUI/SSH stalled")
                except Exception as exc:
                    errors.append(exc)
                    app.destroy()
                    return
                app.after(25, observe)

            try:
                app.after(25, observe)
                app.run_cmds_on_selected()
                app.mainloop()
            finally:
                if app.tk.call("info", "commands", "."):
                    app.destroy()
            self.assertFalse(errors, errors)

    def test_buffered_python_emits_samples_before_process_exit(self):
        # No flush=True: the launcher's -u must guarantee immediate output.
        script = (
            "import sys, time\n"
            "print('startup warning', file=sys.stderr)\n"
            "for i in range(8):\n"
            " print(f'DATA,{i / 5},0,0,1,1,0,0,0,0')\n"
            " time.sleep(0.15)\n"
            "time.sleep(0.5)\n"
        )
        harness = PlotHarness()
        errors = []
        with local_ssh_client() as (client, server):
            def receive():
                try:
                    rc = harness._execute_remote_command(
                        client, "vermelho", f"python3 -c {shlex.quote(script)}", "/tmp",
                    )
                    self.assertEqual(rc, 0)
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=receive)
            started_at = time.monotonic()
            worker.start()
            try:
                while harness._plot_stats["valid_samples"] < 2:
                    self.assertFalse(server.finished.is_set(), "Samples only arrived at exit")
                    self.assertLess(time.monotonic() - started_at, 3, "No live samples")
                    time.sleep(0.01)
                harness.drain_callbacks()
                harness._plot_tick()
                self.assertFalse(server.finished.is_set(), "Plot appeared only after exit")
                self.assertGreater(harness._plot_stats["live_frames"], 0)
                self.assertEqual(len(harness.ax.lines), 2)
                self.assertGreater(len(harness.ax.lines[0].get_ydata()), 1)
                self.assertTrue(harness._active_stream["ready"])
            finally:
                worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors, errors)
            harness.drain_callbacks()
            self.assertEqual(harness._plot_stats["valid_samples"], 8)
            self.assertEqual(harness._plot_stats["invalid_samples"], 0)
            self.assertIn("[VERMELHO] startup warning", harness.terminal)

    def test_fragmented_lines_utf8_crlf_and_eof(self):
        class Packets:
            def __init__(self):
                self.packets = iter([
                    b"DA", b"TA,0,0,0,1,1,0,0,0,0\r", b"\nAcelera\xc3",
                    b"\xa7\xc3\xa3o\rDATA,1,0,0,1,1,0,0,0,0\nlast line", b"",
                ])

            def settimeout(self, timeout):
                pass

            def recv(self, size):
                return next(self.packets)

        lines = list(gui.iter_channel_lines(Packets()))
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[1], "Acelera\u00e7\u00e3o")
        self.assertEqual(gui.parse_telemetry_line(lines[2])["t"], 1)
        self.assertEqual(lines[-1], "last line")

    def test_channel_silence_does_not_end_stream(self):
        class DelayedPackets:
            count = 0

            def settimeout(self, timeout):
                pass

            def recv(self, size):
                self.count += 1
                if self.count == 1:
                    raise socket.timeout()
                return b"hello\n" if self.count == 2 else b""

        idle_events = []
        self.assertEqual(
            list(gui.iter_channel_lines(DelayedPackets(), on_idle=idle_events.append)),
            ["hello"],
        )
        self.assertTrue(idle_events)

    def test_render_failure_retries_and_recovers(self):
        harness = PlotHarness()
        sample = gui.parse_telemetry_line("DATA,0,0,0,1,1,0,0,0,0")
        harness._store_telemetry_sample("vermelho", sample)
        real_draw = harness.canvas.draw
        harness.canvas.draw = lambda: (_ for _ in ()).throw(RuntimeError("draw failed"))
        harness._plot_tick()
        self.assertTrue(harness._plot_dirty)
        self.assertEqual(harness._plot_stats["render_errors"], 1)
        self.assertEqual(len(harness.scheduled), 1)
        harness.canvas.draw = real_draw
        harness._plot_tick()
        self.assertFalse(harness._plot_dirty)
        self.assertIsNone(harness._plot_last_error)

    def test_launcher_quotes_directory_and_prevents_pkill_self_match(self):
        wrapped = gui.build_remote_command('pkill -f "python3.*main.py"', "/tmp/a b'c")
        self.assertIn("[p]ython3.*main.py", wrapped)
        self.assertNotIn('"python3.*main.py"', wrapped)
        wrapped = gui.build_remote_command("python3 main.py", "/tmp")
        self.assertIn("python3 -u main.py", wrapped)
        self.assertIn(gui.STREAM_READY_MARKER, wrapped)
        self.assertNotIn("-u -u", gui.build_remote_command("python3 -u main.py", "/tmp"))


if __name__ == "__main__":
    unittest.main()
