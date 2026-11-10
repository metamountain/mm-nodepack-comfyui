"""mm-nodepack-comfyui: load an image with its path and save edits back next to the source. V3 node API."""
from comfy_api.latest import ComfyExtension

from .mm_nodepack import NODE_LIST, register_routes

WEB_DIRECTORY = "./web"

register_routes()


class MMNodepackExtension(ComfyExtension):
    async def get_node_list(self):
        return NODE_LIST


async def comfy_entrypoint():
    return MMNodepackExtension()
