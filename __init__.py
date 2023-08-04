from .install import install
install()

from .FaceSwapNode import FaceSwapNode

NODE_CLASS_MAPPINGS = {
    "FaceSwapNode": FaceSwapNode,
}

