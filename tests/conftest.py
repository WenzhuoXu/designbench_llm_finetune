"""
pytest configuration for llm_finetune tests.
Adds project root and DesignBench to sys.path.
"""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add DesignBench (for FEA tests)
designbench_path = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
if designbench_path.exists():
    sys.path.insert(0, str(designbench_path))
