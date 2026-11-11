"""Versao do codigo do carro: hash curto dos .py que rodam (fva_car, main.py, tools/).

Gravado no log de cada ensaio e comparado pela GUI antes de executar: nos
ensaios de 10/11 o carro rodou codigo de dias antes e nada no log dizia isso.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_files(root=ROOT):
    root = Path(root)
    files = [root / "main.py"] + sorted((root / "fva_car").glob("*.py")) + sorted((root / "tools").glob("*.py"))
    return [path for path in files if path.is_file()]


def code_version(root=ROOT):
    """8 hexadecimais; muda com qualquer alteracao de codigo (nao com logs ou dados)."""
    digest = hashlib.sha1()
    root = Path(root)
    for path in code_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()[:8]


if __name__ == "__main__":
    print(code_version())
