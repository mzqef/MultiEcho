"""multiecho-ui: a local workbench for running MultiEcho experiments and inspecting their sealed reports."""
from .server import main, serve
from .workspace import Workspace

__version__ = "0.2.0.dev1"
__all__ = ["Workspace", "__version__", "main", "serve"]
