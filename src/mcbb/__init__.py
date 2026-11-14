from . import blanket

# Everything else imports openmc (or jax-sn), so it is imported on first access: `import mcbb.blanket` and
# `mcbb.examples.hcpb()` work without openmc.
_LAZY = ("examples", "materials", "plotting", "openmc_base_functions", "linear_blanket", "sn")


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'mcbb' has no attribute {name!r}")


__all__ = ["blanket", *_LAZY]
