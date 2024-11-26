# # only import if running as a custom node
# try:
# 	pass
# except ImportError:
# 	pass
# else:
# 	NODE_CLASS_MAPPINGS = {}

# # Image Generation
#   ## Sana
# 	from .Image_Generation.Sana.nodes import NODE_CLASS_MAPPINGS as Sana_Nodes
# 	NODE_CLASS_MAPPINGS.update(Sana_Nodes)
 
# NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
from .Image_Generation.Sana.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]