import sys
from pathlib import Path

# Test modules share numeric doubles by importing one another.
sys.path.insert(0, str(Path(__file__).parent))
