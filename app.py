"""
BioCellAgents - Enterprise Digital Pathology & Spatial Biology Platform
=======================================================================
Restored entry point.

The original "yesterday" application source (biocell_app.py) was lost from
disk, but its compiled bytecode (biocell_app.pyc) survived in __pycache__.
This thin wrapper loads that compiled module and runs it, restoring the exact
visual state and behaviour the app had before.

Run: streamlit run app.py
"""
import importlib.machinery
import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_PYC_PATH = _PROJECT_ROOT / "biocell_app.pyc"

if not _PYC_PATH.exists():
    raise FileNotFoundError(
        f"Restored application bytecode not found at: {_PYC_PATH}\n"
        "Re-copy it from __pycache__/biocell_app.cpython-311.pyc"
    )

# Load the compiled "yesterday" module (its top-level code runs st.set_page_config
# and st.markdown(CSS), exactly as the original script did).
_loader = importlib.machinery.SourcelessFileLoader("biocell_app", str(_PYC_PATH))
_spec = importlib.util.spec_from_loader("biocell_app", _loader)
_mod = importlib.util.module_from_spec(_spec)
_loader.exec_module(_mod)

# The original script guarded its UI with `if __name__ == "__main__": main()`.
# When loaded as a module that guard is False, so we invoke main() explicitly.
if hasattr(_mod, "main"):
    _mod.main()
