import sys
from typing import Literal


PlatformType = Literal['macos', 'linux', 'win']
def get_platform() -> PlatformType:
    if sys.platform.startswith('darwin'):
        return "macos"
    elif sys.platform.startswith('win'):
        return "win"
    elif sys.platform.startswith('linux'):
        return "linux"
    else:
        raise Exception(f"Unsupported platform {sys.platform}")


