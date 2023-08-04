from .FaceSwapNode import FaceSwapNode
from .install import install

NODE_CLASS_MAPPINGS = {
    "FaceSwapNode": FaceSwapNode,
}

install()