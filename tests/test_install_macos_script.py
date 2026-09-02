import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "install-macos.sh"


def _run_gate(tmp_path, *, system="Darwin", machine="arm64", translated="0", available_kib=8 * 1024 * 1024):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    commands = {
        "uname": f'#!/bin/sh\n[ "$1" = "-s" ] && echo {system} || echo {machine}\n',
        "sysctl": f"#!/bin/sh\necho {translated}\n",
        "df": f"#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\nmock 1 1 {available_kib} 1%% /\\n'\n",
    }
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "S2S_INSTALL_TEST_ONLY": "1",
    }
    return subprocess.run(["/bin/sh", str(SCRIPT)], capture_output=True, text=True, env=environment)


def test_installer_gates_on_native_apple_silicon_and_disk(tmp_path):
    assert _run_gate(tmp_path).returncode == 0
    assert _run_gate(tmp_path / "linux", system="Linux").returncode != 0
    assert _run_gate(tmp_path / "intel", machine="x86_64").returncode != 0
    assert _run_gate(tmp_path / "rosetta", translated="1").returncode != 0
    disk = _run_gate(tmp_path / "disk", available_kib=3 * 1024 * 1024)
    assert disk.returncode != 0
    assert "4 GiB" in disk.stderr


def test_installer_pins_environment_and_exact_package():
    script = SCRIPT.read_text()

    assert 'SPEECH_TO_SPEECH_VERSION="0.2.13"' in script
    assert 'UV_VERSION="0.11.30"' in script
    assert 'PYTHON_VERSION="3.11.13"' in script
    assert 'UV_SHA256="2b9e582af54f84fa50c115427451a6c13e80f43b52f8282b8af5791077317bbf"' in script
    assert "macos-arm64-constraints.txt" in script
    assert "speech-to-speech==${SPEECH_TO_SPEECH_VERSION}" in script
    assert "--constraints" in script


def test_installer_has_valid_posix_shell_syntax():
    assert subprocess.run(["/bin/sh", "-n", str(SCRIPT)]).returncode == 0
