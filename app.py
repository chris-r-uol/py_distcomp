"""Convenience wrapper so ``streamlit run app.py`` still works in a checkout.

The demo itself lives in :mod:`py_distcomp.demo` so that it ships with the
package; installed users can launch it with ``py-distcomp-demo``.
"""

from py_distcomp.demo import main

if __name__ == "__main__":
    main()
