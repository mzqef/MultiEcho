"""multiecho-ui: a local, read-only web viewer for MultiEcho reports, studies and keyframe galleries."""
from .server import main, serve
from .workspace import Workspace

__version__ = "0.1.0"
__all__ = ["Workspace", "__version__", "main", "serve"]
