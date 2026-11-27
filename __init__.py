"""mm-nodepack-comfyui: load an image with its path, save edits back next to the source,
upscale/crop with Upscale Crop MM, and call vision / text LLMs with LLM Multitool MM.
V3 node API."""
import sys

WEB_DIRECTORY = "./web"

if "pytest" not in sys.modules:
    from comfy_api.latest import ComfyExtension

    from .py.image_io import NODE_LIST, register_routes
    from .py.upscale_crop import NODE_LIST as UPSCALE_NODES, register_routes as register_upscale_routes
    from .py.llm_multitool import NODE_LIST as LLM_NODES, register_routes as register_llm_routes

    register_routes()
    register_upscale_routes()
    register_llm_routes()


    class MMNodepackExtension(ComfyExtension):
        async def get_node_list(self):
            return NODE_LIST + UPSCALE_NODES + LLM_NODES


    async def comfy_entrypoint():
        return MMNodepackExtension()
