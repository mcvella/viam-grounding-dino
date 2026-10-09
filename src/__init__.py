"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .groundingDino import groundingDino

Registry.register_resource_creator(Vision.API, groundingDino.MODEL, ResourceCreatorRegistration(groundingDino.new, groundingDino.validate))
