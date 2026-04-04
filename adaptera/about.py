"""
Adaptera - System Information & Diagnostics
"""

from importlib.metadata import version, PackageNotFoundError
import importlib.util
import platform


# -----------------------------
# Metadata
# -----------------------------
PROJECT_NAME = "Adaptera"
REPO_URL = "https://github.com/Sylo3285/Adaptera"
DOCS_URL = "https://github.com/Sylo3285/Adaptera#readme"
YOUTUBE_URL = "https://www.youtube.com/@Sylosoft"
DISCORD_URL = "https://discord.gg/p4JeXKVxSB"
LICENSE = "MIT"

MODULES = {
    "torch": "PyTorch backend",
    "transformers": "Hugging Face Transformers",
    "peft": "Parameter-Efficient Fine-Tuning",
    "bitsandbytes": "QLoRA 8-bit/4-bit support",
    "accelerate": "HF Accelerate",
    "datasets": "HF Datasets",
    "faiss": "FAISS vector database",
}


# -----------------------------
# Helpers
# -----------------------------
def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _get_version_safe(pkg: str):
    try:
        return version(pkg)
    except Exception:
        return None


# -----------------------------
# RAM / VRAM
# -----------------------------
def _get_ram():
    try:
        import psutil
        total = psutil.virtual_memory().total
        return round(total / (1024**3))  # GB
    except Exception:
        return None


def _get_vram():
    if not _is_installed("torch"):
        return None

    try:
        import torch

        if not torch.cuda.is_available():
            return None

        total = torch.cuda.get_device_properties(0).total_memory
        return round(total / (1024**3))  # GB
    except Exception:
        return None


# -----------------------------
# Torch Info
# -----------------------------
def _torch_info():
    if not _is_installed("torch"):
        return {"installed": False}

    import torch

    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": None,
        "device_count": 0,
    }

    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        try:
            info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu"] = "Unknown"

    return info


# -----------------------------
# Dependency Info
# -----------------------------
def _dependency_info():
    deps = {}

    for module in MODULES:
        if module == "torch":
            deps["torch"] = _torch_info()
            continue

        installed = _is_installed(module)
        deps[module] = {
            "installed": installed,
            "version": _get_version_safe(module) if installed else None,
        }

    return deps


# -----------------------------
# Diagnostics
# -----------------------------
def _run_diagnostics(deps: dict):
    issues = []
    warnings = []

    torch = deps.get("torch", {})

    if torch.get("installed"):
        if not torch.get("cuda_available"):
            warnings.append("CUDA not available (running on CPU)")
    else:
        issues.append("PyTorch not installed")

    if deps.get("bitsandbytes", {}).get("installed"):
        try:
            import bitsandbytes  # noqa
        except Exception:
            issues.append("bitsandbytes installed but failed to import")

    if deps.get("accelerate", {}).get("installed"):
        try:
            from accelerate.state import AcceleratorState  # noqa
        except Exception:
            warnings.append("accelerate installed but not configured")

    return {
        "issues": issues,
        "warnings": warnings,
    }


# -----------------------------
# Collect Info
# -----------------------------
def collect_info():
    deps = _dependency_info()

    info = {
        "project": {
            "name": PROJECT_NAME,
            "version": _get_version_safe("adaptera") or "unknown (dev)",
            "license": LICENSE,
            "repo": REPO_URL,
            "docs": DOCS_URL,
        },
        "system": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "ram": _get_ram(),
            "vram": _get_vram(),
        },
        "dependencies": deps,
    }

    info["diagnostics"] = _run_diagnostics(deps)

    return info


# -----------------------------
# Formatting
# -----------------------------
def _format_dependency(name, data):
    if name == "torch":
        if not data["installed"]:
            return "Not Installed"

        cuda = "Yes" if data["cuda_available"] else "No"
        gpu = f", GPU: {data['gpu']}" if data["gpu"] else ""

        return f"{data['version']} (CUDA: {cuda}{gpu})"

    if not data["installed"]:
        return "Not Installed"

    return data["version"] or "Installed"


def _print_info(info: dict):
    print("=" * 60)
    print(f"{info['project']['name']} v{info['project']['version']}")
    print("Local-first LLM orchestration framework")
    print("=" * 60)
    print()

    # System
    print("System")
    print("-" * 60)
    print(f"Python   : {info['system']['python']}")
    print(f"Platform : {info['system']['platform']}")

    ram = info["system"].get("ram")
    vram = info["system"].get("vram")

    print(f"RAM      : {ram} GB" if ram else "RAM      : Unknown")
    print(f"VRAM     : {vram} GB" if vram else "VRAM     : None / Unknown")
    print()

    # Dependencies
    print("Core Dependencies")
    print("-" * 60)
    for name, data in info["dependencies"].items():
        print(f"{name:<15}: {_format_dependency(name, data)}")
    print()

    # Diagnostics
    print("Diagnostics")
    print("-" * 60)

    diag = info["diagnostics"]

    if not diag["issues"] and not diag["warnings"]:
        print("✓ No issues detected")

    for w in diag["warnings"]:
        print(f"⚠ {w}")

    for i in diag["issues"]:
        print(f"✗ {i}")

    print()

    # Project
    print("Project")
    print("-" * 60)
    print(f"License      : {info['project']['license']}")
    print(f"Repository   : {info['project']['repo']}")
    print(f"Documentation: {info['project']['docs']}")
    print()

    # Community
    print("Community")
    print("-" * 60)
    print(f"Youtube      : {YOUTUBE_URL}")
    print(f"Discord      : {DISCORD_URL}")

    print("=" * 60)


# -----------------------------
# Public API
# -----------------------------
def about():
    info = collect_info()
    _print_info(info)


if __name__ == "__main__":
    about()