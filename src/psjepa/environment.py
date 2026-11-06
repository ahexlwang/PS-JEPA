'Resolve PS-JEPA environment settings.'
import os

def getenv(name: str, default=None):
    if name in os.environ:
        return os.environ[name]
    return os.environ.get(name.replace("PSJEPA_", "BIOJEPA_", 1), default)
