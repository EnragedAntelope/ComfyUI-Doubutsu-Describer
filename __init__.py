from .nodes.doubutsu_describer import DoubutsuDescriber

NODE_CLASS_MAPPINGS = {
    "DoubutsuDescriber": DoubutsuDescriber
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubutsuDescriber": "Doubutsu Image Describer"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]