import insightface
import onnxruntime
import torch
import glob
import tempfile
import numpy as np
import cv2
import os
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
import folder_paths
import torchvision.transforms as T
from comfy import model_management

providers = ["CPUExecutionProvider"]
model_path = folder_paths.models_dir
onnx_path = os.path.join(model_path, "roop")
FS_MODEL = None
CURRENT_FS_MODEL_PATH = None
device = model_management.get_torch_device()


def get_models():
    models_path = os.path.join(onnx_path + os.path.sep + "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models


def convert_to_sd(img):
    return [False, tempfile.NamedTemporaryFile(delete=False, suffix=".png")]


class FaceSwapNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"face": ("IMAGE",),
                             "image": ("IMAGE",),
                             "source_face_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                             "target_face_indices": ("STRING", {"multiline": False}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"

    CATEGORY = "image/faceswap"

    def swap(self, face: torch.Tensor, image: torch.Tensor, source_face_index=0, target_face_indices="0"):
        models = get_models()

        target_faces = {int(x) for x in target_face_indices.strip(",").split(",") if x.isnumeric()}
        result = swap_face(face, image, models[0], source_face_index, target_faces)

        result_tensor = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_tensor)[None,]

        return (result_tensor,)


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
    return FS_MODEL


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)

    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def swap_face(
        source_img: torch.Tensor,
        target_img: torch.Tensor,
        model: Union[str, None] = None,
        source_face: [int] = 0,
        target_face_list: Set[int] = {0},
) -> Image.Image:
    result_image = target_img
    converted = convert_to_sd(target_img)
    scale, fn = converted[0], converted[1]
    if model is not None and not scale:

        source_img = (source_img[0].detach().numpy() * 255).astype(np.uint8)
        target_img = (target_img[0].detach().numpy() * 255).astype(np.uint8)

        source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=source_face)

        if source_face is not None:
            result = target_img
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            face_swapper = getFaceSwapModel(model_path)

            for face_num in target_face_list:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                    result_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result_image
