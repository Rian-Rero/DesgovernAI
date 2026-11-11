"""Coleta incremental, limpeza do carro e relogio da fva_gui_um_carro.

O "carro" e uma pasta temporaria: o cliente SSH falso roda os comandos num shell
de verdade (find, tar, xargs, df, date), como no Raspberry."""
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fva_gui_coleta", ROOT / "tools" / "fva_gui_um_carro.py")
gui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gui)


class _Out:
    def __init__(self, process):
        self.process = process
        self.channel = SimpleNamespace(recv_exit_status=process.wait)

    def read(self, size=-1):
        return self.process.stdout.read() if size is None or size < 0 else self.process.stdout.read(size)


class _Err:
    def __init__(self, process):
        self.process = process

    def read(self, size=-1):
        return self.process.stderr.read()


class _In:
    def __init__(self, process):
        self.process = process
        self.written = b""
        self.channel = SimpleNamespace(shutdown_write=process.stdin.close)

    def write(self, text):
        data = text.encode() if isinstance(text, str) else text
        self.written += data
        self.process.stdin.write(data)

    def flush(self):
        self.process.stdin.flush()


class ShellClient:
    """exec_command num shell local (cwd qualquer: os comandos fazem cd)."""
    def __init__(self, replace=None):
        self.commands, self.inputs, self.replace = [], [], replace or {}
        self.processes = []

    def close(self):
        for process in self.processes:
            for stream in (process.stdin, process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()
            process.wait()

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        for word, substitute in self.replace.items():
            if word in command:
                command = substitute
        process = subprocess.Popen(["bash", "-c", command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.processes.append(process)
        stdin = _In(process)
        self.inputs.append(stdin)
        return stdin, _Out(process), _Err(process)


class FakeSftp:
    def get(self, remote, local):
        shutil.copyfile(remote, local)


def write(path, data, age=3600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


class ListingTests(unittest.TestCase):
    def test_parse_rejects_paths_outside_the_folder(self):
        files = gui.parse_remote_listing("a/car.csv\t10\t5.5\n../x\t1\t1\nlixo\n/etc/passwd\t1\t1\nb c\t3\t2\n")
        self.assertEqual(files, {"a/car.csv": (10, 5.5), "b c": (3, 2.)})

    def test_plan_skips_what_is_on_the_pc_or_was_collected_before(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            write(local / "r1" / "car.csv", b"12345")
            remote = {"r1/car.csv": (5, 100.), "r2/car.csv": (7, 200.), "r3/car.csv": (9, 300.),
                      "r1/cones_debug.jsonl": (4, 100.)}
            manifest = {"r2/car.csv": [7, 200]}  # baixado antes e movido no PC
            self.assertEqual(gui.collect_plan(remote, folder, manifest), ["r1/cones_debug.jsonl", "r3/car.csv"])
            self.assertEqual(len(gui.collect_plan(remote, folder, manifest, force=True)), 4)
            remote["r1/car.csv"] = (6, 150.)  # cresceu no carro: baixa de novo
            self.assertIn("r1/car.csv", gui.collect_plan(remote, folder, manifest))

    def test_only_collected_old_files_can_be_deleted_and_never_the_trim(self):
        with tempfile.TemporaryDirectory() as folder:
            remote = {"r1/car.csv": (5, 1000.), "r2/car.csv": (5, 1000.), "agora/car.csv": (5, 1990.),
                      "direcao_trim.json": (5, 1000.)}
            manifest = {rel: [5, int(info[1])] for rel, info in remote.items() if rel != "r2/car.csv"}
            self.assertEqual(gui.deletable(remote, folder, manifest, remote_now=2000.), ["r1/car.csv"])
            self.assertEqual(gui.deletable(remote, folder, manifest, remote_now=None), [])


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.car, self.pc = base / "carro" / "logs", base / "pc"
        self.pc.mkdir()
        write(self.car / "20261111_100000" / "car.csv", b"t,x\n0,0\n")
        for k in range(12):
            write(self.car / "20261111_100000" / "frames" / f"f{k:06d}.jpg", bytes([k]) * (100 + k))
        write(self.car / "terminal" / "missao com espaco.log", b"ok\n")
        write(self.car / "direcao_trim.json", b'{"steering_trim_deg": 1.0}')
        write(self.car / "20261111_110000" / "car.csv", b"recente", age=5)

    def tearDown(self):
        self.temp.cleanup()

    def collect(self, client, force=False):
        remote, now = gui.remote_listing(client, str(self.car))
        manifest = gui.load_manifest(str(self.pc))
        todo = gui.collect_plan(remote, str(self.pc), manifest, force)
        messages = []
        done = gui.RsyncGUI._download_files(None, client, FakeSftp(), str(self.car), str(self.pc), remote, todo,
                                            manifest, messages.append)
        gui.save_manifest(str(self.pc), manifest)
        client.close()
        return remote, now, manifest, done, messages

    def test_first_collect_brings_everything_and_the_second_nothing(self):
        client = ShellClient()
        remote, now, _, done, messages = self.collect(client)
        self.assertEqual(done, len(remote))
        self.assertEqual(len(remote), 16)
        self.assertTrue(any("tar -cf" in command for command in client.commands))  # muitos arquivos: tar
        self.assertFalse(any("interrompido" in message for message in messages))
        for rel, (size, mtime) in remote.items():
            path = self.pc / rel
            self.assertEqual(path.read_bytes(), (self.car / rel).read_bytes(), rel)
            self.assertEqual(int(path.stat().st_mtime), int(mtime))
        self.assertFalse(list(self.pc.rglob("*.part")))
        self.assertAlmostEqual(now, time.time(), delta=5)
        _, _, _, again, _ = self.collect(client)
        self.assertEqual(again, 0)

    def test_moving_runs_on_the_pc_does_not_download_them_again(self):
        self.collect(ShellClient())
        (self.pc / "organizados").mkdir()
        shutil.move(str(self.pc / "20261111_100000"), str(self.pc / "organizados"))
        _, _, _, again, _ = self.collect(ShellClient())
        self.assertEqual(again, 0)

    def test_broken_tar_falls_back_to_sftp(self):
        client = ShellClient(replace={"tar -cf": "echo nada"})
        remote, _, _, done, messages = self.collect(client)
        self.assertEqual(done, len(remote))
        self.assertTrue(any("interrompido" in message for message in messages))

    def test_tar_member_outside_the_request_is_ignored(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            for name in ("../fora.txt", "a/pedido.txt"):
                info = tarfile.TarInfo(name)
                info.size = 2
                archive.addfile(info, io.BytesIO(b"xx"))
        payload.seek(0)
        stream = SimpleNamespace(read=payload.read)
        client = SimpleNamespace(exec_command=lambda command, timeout=None: (
            SimpleNamespace(write=lambda text: None, flush=lambda: None,
                            channel=SimpleNamespace(shutdown_write=lambda: None)), stream, None))
        todo = [f"a/pedido.txt"] + [f"a/{k}" for k in range(8)]
        remote = {rel: (2, 0.) for rel in todo}
        wanted = []

        class Sftp:
            def get(self, remote_path, local):
                wanted.append(remote_path)
                Path(local).write_bytes(b"yy")

        gui.RsyncGUI._download_files(None, client, Sftp(), "/carro", str(self.pc), remote, todo, {}, lambda m: None)
        self.assertEqual((self.pc / "a" / "pedido.txt").read_bytes(), b"xx")
        self.assertFalse((Path(self.temp.name) / "fora.txt").exists())
        self.assertEqual(len(wanted), 8)  # o resto veio por SFTP

    def test_delete_removes_only_collected_files_and_keeps_trim_and_recent(self):
        remote, now, manifest, _, _ = self.collect(ShellClient())
        candidates = gui.deletable(remote, str(self.pc), manifest, now)
        self.assertNotIn("direcao_trim.json", candidates)
        self.assertNotIn("20261111_110000/car.csv", candidates)
        self.assertEqual(len(candidates), 14)
        messages = []
        app = SimpleNamespace(_connect_ssh=lambda ip: ShellClient(),
                              ui=lambda function, text: messages.append(text), datalog_write=None)
        gui.RsyncGUI._delete_remote(app, "carro", "1.2.3.4", str(self.car), candidates, 1000)
        left = sorted(str(p.relative_to(self.car)) for p in self.car.rglob("*") if p.is_file())
        self.assertEqual(left, ["20261111_110000/car.csv", "direcao_trim.json"])
        self.assertFalse((self.car / "20261111_100000").exists())  # pasta vazia sai junto
        self.assertTrue(any("Apagados 14 arquivos" in message for message in messages), messages)

    def test_missing_logs_folder_is_reported(self):
        client = ShellClient()
        self.assertEqual(gui.remote_listing(client, str(self.car / "nao_existe")), (None, None))
        client.close()


class ClockTests(unittest.TestCase):
    def test_command_uses_sudo_with_the_password_on_stdin(self):
        command, secret = gui.clock_command(1762870000.4, "senha", "America/Sao_Paulo")
        self.assertTrue(command.startswith("sudo -S -p ''"))
        self.assertIn("date -s @1762870000", command)
        self.assertIn("timedatectl set-timezone America/Sao_Paulo", command)
        self.assertIn("fake-hwclock save", command)
        self.assertNotIn("senha", command)  # nunca na linha de comando (ps)
        self.assertEqual(secret, "senha\n")
        command, secret = gui.clock_command(1762870000, None, "x; rm -rf /")
        self.assertTrue(command.startswith("sudo -n"))
        self.assertNotIn("rm -rf", command)
        self.assertEqual(secret, "")

    def run_sync(self, offset_s, zone=None, force=False):
        state = {"offset": offset_s}
        zone = zone or time.strftime("%z")
        sent = []

        def date_command(command, timeout=None):
            if command.startswith("date +%s.%N"):
                text = f"{time.time() + state['offset']:.3f}\n{zone}\n"
                return None, SimpleNamespace(read=lambda: text.encode()), None
            sent.append(command)
            state["offset"] = 0.  # o sudo acertou
            stdin = SimpleNamespace(write=lambda text: sent.append(("stdin", text)), flush=lambda: None,
                                    channel=SimpleNamespace(shutdown_write=lambda: None))
            out = SimpleNamespace(read=lambda: b"", channel=SimpleNamespace(recv_exit_status=lambda: 0))
            return stdin, out, SimpleNamespace(read=lambda: b"")

        messages = []
        ok = gui.sync_car_clock(SimpleNamespace(exec_command=date_command), "senha", messages.append, force)
        return ok, sent, messages

    def test_right_clock_is_left_alone(self):
        ok, sent, messages = self.run_sync(.3)
        self.assertTrue(ok)
        self.assertEqual(sent, [])
        self.assertEqual(messages, [])

    def test_late_clock_is_set_with_the_password(self):
        ok, sent, messages = self.run_sync(-18*86400.)
        self.assertTrue(ok)
        self.assertIn(("stdin", "senha\n"), sent)
        self.assertIn("atrasado 18.0 dias", messages[0])
        self.assertIn("acertado", messages[-1])


if __name__ == "__main__":
    unittest.main()
