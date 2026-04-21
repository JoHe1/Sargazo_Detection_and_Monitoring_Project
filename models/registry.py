"""
models/registry.py
--------------------
Registro central de modelos del proyecto.

Permite instanciar cualquier modelo por nombre desde ExperimentConfig
sin importar directamente la clase en train.py o inference.py.

Uso:
    from models.registry import ModelRegistry

    # Instanciar por nombre
    model = ModelRegistry.build("swin_transformer", num_classes=16)
    model = ModelRegistry.build("segformer", num_classes=16)

    # Listar modelos disponibles
    print(ModelRegistry.available())

Para añadir un nuevo modelo al proyecto:
    1. Crea su archivo en models/architectures/
    2. Importa la clase aquí
    3. Regístrala con @ModelRegistry.register("nombre")
       o añádela al dict REGISTRY manualmente
"""

from __future__ import annotations

from typing import Type

import torch.nn as nn


# ── Importar todas las arquitecturas disponibles ───────────────────────
from models.architectures.swin_transformer import SwinSegmenter
from models.architectures.segformer import SegFormerSegmenter
from models.architectures.swin_transformer_attention import SwinAttSegmenter


class ModelRegistry:
    """
    Registro central que mapea nombre → clase de modelo.

    El propósito es poder escribir en ExperimentConfig:
        model_name = "swin_transformer"
    Y que el trainer instancie el modelo correcto sin importar
    directamente la clase en cada script.
    """

    # Dict principal: nombre_en_config → clase
    _REGISTRY: dict[str, Type[nn.Module]] = {
        "swin_transformer": SwinSegmenter,
        "swin_transformer_attention": SwinAttSegmenter,
        "segformer":        SegFormerSegmenter,
    }

    # ── API pública ───────────────────────────────────────────────────

    @classmethod
    def build(cls, model_name: str, **kwargs) -> nn.Module:
        """
        Instancia un modelo por nombre.

        Args:
            model_name: nombre registrado del modelo (ej: "swin_transformer")
            **kwargs:   argumentos pasados al constructor del modelo
                        (ej: num_classes=16)

        Returns:
            instancia del modelo lista para usar

        Raises:
            ValueError si el nombre no está registrado
        """
        if model_name not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Modelo '{model_name}' no registrado.\n"
                f"Disponibles: {available}"
            )
        model_class = cls._REGISTRY[model_name]
        return model_class(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Devuelve la lista de nombres de modelos registrados."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def register(cls, name: str):
        """
        Decorador para registrar una nueva arquitectura.

        Uso:
            @ModelRegistry.register("mi_modelo")
            class MiModelo(BaseModel):
                ...
        """
        def decorator(model_class: Type[nn.Module]):
            cls._REGISTRY[name] = model_class
            return model_class
        return decorator