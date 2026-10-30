"""Assemble a deployable copy of the pocketing UI: Google Cloud Run or a Hugging Face Space.

Copies the frcpocket package and the trained models into a self-contained folder with
app.py and pinned requirements. Cloud Run also gets a Dockerfile (Cloud Build turns it
into the container, so Docker isn't needed locally); a Space gets packages.txt and the
Space settings in README.md. Rerun after any change, then deploy the folder.

    .venv/bin/python -I deploy/build.py --target cloudrun     # -> deploy/cloudrun
    gcloud run deploy frc-pocketing --source deploy/cloudrun ...   (see the README)
    .venv/bin/python -I deploy/build.py --target space        # -> deploy/space
    hf upload <user>/<space> deploy/space . --repo-type space
"""

import argparse
import shutil
from pathlib import Path

repoRoot = Path(__file__).resolve().parent.parent
# The versions every test here ran on; Spaces installs these (gradio comes from sdk_version).
pythonVersion = "3.12"
gradioVersion = "6.31.0"
requirements = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.15.0
cadquery==2.8.0
cadquery-ocp==7.9.3.1.1
scikit-fem==12.0.2
shapely==2.2.0
numpy==2.5.4
scipy==1.18.1
matplotlib==3.11.2
contourpy==1.4.0
pillow==12.3.0
ezdxf==1.4.5
triangle==20250106
"""
# OpenCascade's viewer libraries load at import even without a display.
systemPackages = """\
libgl1
libglu1-mesa
libxrender1
"""
appEntry = '''"""Hosted entry point for the FRC plate pocketing UI."""

import os

from gradio import networking

from frcpocket.app import buildApp

# Gradio pings itself at http://localhost:<port>/ after starting and refuses to run if
# that fails. Inside Cloud Run's sandbox the ping fails even though the app serves fine
# on 0.0.0.0, and Cloud Run health-checks the port itself, so skip the ping there.
if os.environ.get("K_SERVICE"):  # set by Cloud Run
    networking.url_ok = lambda url: True

# Only the main process starts the site: the FEA worker processes are spawned, and a
# spawned process re-imports this file. No host or port here: the host sets them through
# GRADIO_SERVER_NAME / GRADIO_SERVER_PORT.
if __name__ == "__main__":
    buildApp().launch()
'''
dockerfile = f"""\
FROM python:{pythonVersion}-slim
# OpenCascade's viewer libraries load at import even without a display; libgomp for torch.
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libglu1-mesa libxrender1 libgomp1 \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio=={gradioVersion}
COPY . .
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False MPLBACKEND=Agg
# Cloud Run passes the port to listen on in $PORT.
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${{PORT:-8080}} exec python app.py"]
"""
ignoreFile = """\
__pycache__/
*.pyc
"""
spaceReadme = f"""---
title: FRC Plate Pocketing
emoji: 🛠️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: {gradioVersion}
python_version: "{pythonVersion}"
app_file: app.py
pinned: false
---

# FRC plate pocketing

Upload a flat FRC plate as STEP, set wall, rib, corner radius and boss ring, and get
ML-generated pocketing candidates that pass every hard rule, then download the pocketed
STEP. Built from the frcpocket repository; rebuild this Space with `deploy/buildSpace.py`.

On the free CPU hardware a plate takes about 1–3 minutes. Thumbs-up/down ratings are
saved to the Space's disk, which is wiped when the Space restarts.
"""
# Not needed by the UI: scrapers and offline data tools.
skippedModules = {"scrape"}


def build(outDir, target="space"):
    outDir = Path(outDir)
    if outDir.exists():
        shutil.rmtree(outDir)
    package = outDir / "frcpocket"
    shutil.copytree(repoRoot / "src" / "frcpocket", package,
                    ignore=lambda folder, names: [n for n in names if n == "__pycache__" or n in skippedModules])
    (outDir / "models").mkdir(parents=True)
    for name in ("generator.pt", "aestheticScorer.pt", "aestheticScorer.json"):
        shutil.copy2(repoRoot / "models" / name, outDir / "models" / name)
    (outDir / "app.py").write_text(appEntry)
    (outDir / "requirements.txt").write_text(requirements)
    if target == "cloudrun":
        (outDir / "Dockerfile").write_text(dockerfile)
        (outDir / ".dockerignore").write_text(ignoreFile)
        (outDir / ".gcloudignore").write_text(ignoreFile)
    else:
        (outDir / "packages.txt").write_text(systemPackages)
        (outDir / "README.md").write_text(spaceReadme)
    files = sorted(path for path in outDir.rglob("*") if path.is_file())
    size = sum(path.stat().st_size for path in files)
    print(f"{outDir}: {len(files)} files, {size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["cloudrun", "space"], default="cloudrun")
    parser.add_argument("--out", default=None, help="default: deploy/<target>")
    args = parser.parse_args()
    build(args.out or repoRoot / "deploy" / args.target, args.target)


if __name__ == "__main__":
    main()
