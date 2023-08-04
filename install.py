import os
import sys
import subprocess

comfy_path = '../..'
if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version

impact_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(impact_path)
sys.path.append(comfy_path)

import platform
import folder_paths
from torchvision.datasets.utils import download_url

print("### ComfyUI-FaceSwapper: Check dependencies")

if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
    pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
else:
    pip_install = [sys.executable, '-m', 'pip', 'install']

def ensure_pip_packages():
    try:
        import cython
    except Exception:
        my_path = os.path.dirname(__file__)
        requirements_path = os.path.join(my_path, "requirements.txt")
        subprocess.check_call(pip_install + ['-r', requirements_path])

def install():
    ensure_pip_packages()
    # Download model
    print("### ComfyUI-Impact-Pack: Check basic models")
    model_path = folder_paths.models_dir
    onnx_path = os.path.join(model_path, "roop")

    if not os.path.exists(onnx_path):
        download_url("https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx", onnx_path)

