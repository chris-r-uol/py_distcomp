"""Console entry point for the Streamlit demo.

A Streamlit app cannot simply be called as a Python function -- it needs the
Streamlit runtime around it -- so the ``py-distcomp-demo`` command re-launches
the demo module through ``streamlit run``.
"""

import pathlib
import sys


def main() -> int:
    """Launch the demo app with ``streamlit run``."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        sys.stderr.write(
            "The demo app needs streamlit, which is an optional extra.\n"
            "Install it with:  pip install 'py_distcomp[app]'\n"
        )
        return 1

    demo = pathlib.Path(__file__).with_name("demo.py")
    sys.argv = ["streamlit", "run", str(demo), *sys.argv[1:]]
    return streamlit_cli.main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
