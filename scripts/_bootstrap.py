"""Put the project root on sys.path so scripts can import src, however they are started."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
