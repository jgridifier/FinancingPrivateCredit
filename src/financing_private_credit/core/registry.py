"""
Generic Registry Pattern

Provides a reusable registry for:
- Indicators
- Models/Optimizers
- Data sources
- Dashboard components
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, Optional

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for named components.

    Usage:
        # Create a registry
        model_registry = Registry[BaseModel]("models")

        # Register via decorator
        @model_registry.register("aplr")
        class APLRModel(BaseModel):
            pass

        # Or register directly
        model_registry.register("ols")(OLSModel)

        # Get a registered class
        model_cls = model_registry.get("aplr")
        model = model_cls(**config)

        # List registered items
        print(model_registry.list())
    """

    def __init__(self, name: str):
        """
        Initialize registry.

        Args:
            name: Name of this registry (for error messages)
        """
        self.name = name
        self._registry: dict[str, type[T]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        aliases: Optional[list[str]] = None,
        **metadata,
    ) -> Callable[[type[T]], type[T]]:
        """
        Register a class with this registry.

        Args:
            name: Primary registration name
            aliases: Optional list of alias names
            **metadata: Additional metadata to store

        Returns:
            Decorator function
        """
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self.name} registry"
                )

            self._registry[name] = cls
            self._metadata[name] = {
                "class": cls,
                "aliases": aliases or [],
                **metadata,
            }

            # Register aliases
            for alias in (aliases or []):
                if alias in self._registry:
                    raise ValueError(
                        f"Alias '{alias}' already registered in {self.name} registry"
                    )
                self._registry[alias] = cls

            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """
        Get a registered class by name.

        Args:
            name: Registered name or alias

        Returns:
            Registered class

        Raises:
            KeyError: If name not found
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def create(self, name: str, **kwargs) -> T:
        """
        Create an instance of a registered class.

        Args:
            name: Registered name
            **kwargs: Arguments to pass to constructor

        Returns:
            Instance of registered class
        """
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list[str]:
        """Return list of primary registered names (excluding aliases)."""
        return [
            name for name, meta in self._metadata.items()
        ]

    def list_all(self) -> list[str]:
        """Return list of all registered names including aliases."""
        return list(self._registry.keys())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a registered item."""
        # Find primary name if alias was provided
        if name not in self._metadata:
            for primary, meta in self._metadata.items():
                if name in meta.get("aliases", []):
                    return meta
            raise KeyError(f"'{name}' not found in {self.name} registry")
        return self._metadata[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._metadata)


# Pre-configured registries for the application
indicator_registry = Registry["BaseIndicator"]("indicators")
model_registry = Registry["BaseModel"]("models")
optimizer_registry = Registry["BaseOptimizer"]("optimizers")
data_source_registry = Registry["BaseDataSource"]("data_sources")
