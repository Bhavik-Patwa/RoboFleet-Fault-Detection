import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

REQUIRED_DIRECTORIES = [
    PROJECT_ROOT / "data",
]


def run_command(command):
    subprocess.run(command, check = True)


def find_virtualenv_python():
    unix_python = VENV_DIR / "bin" / "python"
    windows_python = VENV_DIR / "Scripts" / "python.exe"

    if unix_python.exists():
        return unix_python

    if windows_python.exists():
        return windows_python

    return None


def ensure_project_files():
    # Failing fast when the tracked dependency file is missing.
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Required file not found : {REQUIREMENTS_FILE}")


def ensure_directories():
    # Creating only the local runtime directories that must exist.
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents = True, exist_ok = True)

    print("Required project directories are ready.")


def create_virtual_environment():
    existing_python = find_virtualenv_python()

    # Reusing the existing virtual environment when it is already available.
    if existing_python is not None:
        print("Virtual environment already exists.")
        return

    print("Creating virtual environment:")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_dependencies():
    python_executable = find_virtualenv_python()

    if python_executable is None:
        raise FileNotFoundError("Virtual environment Python executable not found.")

    print("Upgrading pip:")
    run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])

    # Installing the tracked project dependencies into the project environment.
    print("Installing dependencies from requirements.txt :")
    run_command([str(python_executable), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def main():
    print(f"Project root: {PROJECT_ROOT}")
    ensure_project_files()
    ensure_directories()
    create_virtual_environment()
    install_dependencies()
    print("Bootstrap completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Bootstrap failed : {exc}")
        sys.exit(1)