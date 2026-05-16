import importlib.metadata

#__version__ = importlib.metadata.version("mem0ai")

try:
    from importlib.metadata import version
    __version__ = version("mem0ai")
except Exception:
    __version__ = "0.1.0" # 给它加个异常处理，或者直接硬编码

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
