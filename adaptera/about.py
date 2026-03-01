"""
Adaptera - Local-first LLM orchestration framework.
Provides information about the installed Adaptera package.
"""

from importlib.metadata import version, PackageNotFoundError
import importlib.util
import platform



PROJECT_NAME = "Adaptera"
REPO_URL = "https://github.com/Sylo3285/Adaptera"
DOCS_URL = "https://github.com/Sylo3285/Adaptera#readme"
YOUTUBE_URL = "https://www.youtube.com/@Sylosoft"
DISCORD_URL = "https://discord.gg/DM76sSz9GR"
LICENSE = "MIT"


# Dependencies
MODULES = {
    "torch": "PyTorch backend",
    "transformers": "Hugging Face Transformers",
    "peft": "Parameter-Efficient Fine-Tuning",
    "bitsandbytes": "QLoRA 8-bit/4-bit support",
    "accelerate": "HF Accelerate",
    "datasets": "HF Datasets",
    "faiss": "FAISS vector database",
}


def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _torch_cuda_status() -> str:
    if not _is_installed("torch"):
        return "Not Installed"

    import torch  # Safe because we checked
    cuda = "Yes" if torch.cuda.is_available() else "No"
    return f"Installed (CUDA: {cuda})"


def get_version() -> str:
    try:
        return version("adaptera")
    except PackageNotFoundError:
        return "unknown (development)"


def about() -> None:
    print("=" * 60)
    print(f"{PROJECT_NAME} v{get_version()}")
    print("Local-first LLM orchestration framework")
    print("=" * 60)
    print()

    # System Info
    print("System")
    print("-" * 60)
    print(f"Python   : {platform.python_version()}")
    print(f"Platform : {platform.system()} {platform.release()}")
    print()

    # Dependencies
    print("Core Dependencies")
    print("-" * 60)

    for module, desc in MODULES.items():
        if module == "torch":
            status = _torch_cuda_status()
        else:
            status = "Installed" if _is_installed(module) else "Not Installed"

        print(f"{module:<15}: {status}")

    print()
    print("Project")
    print("-" * 60)
    print(f"License      : {LICENSE}")
    print(f"Repository   : {REPO_URL}")
    print(f"Documentation: {DOCS_URL}")

    print()
    print("Community")
    print("-" * 60)
    print(f"Youtube      : {YOUTUBE_URL}")
    print(f"Discord      : {DISCORD_URL}")

    print("=" * 60)

if __name__ == "__main__":
    about()