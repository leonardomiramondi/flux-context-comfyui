from .flux_replicate_node import FluxReplicateNode

NODE_CLASS_MAPPINGS = {
    "FluxReplicateNode": FluxReplicateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxReplicateNode": "Flux Replicate Context"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 