import sys
import os
from pathlib import Path

# Add the project root to sys.path so 'src' can be imported everywhere
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
