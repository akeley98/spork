"""``python3 -m code_to_tex SRC OUTDIR``: run the listing generator."""

import sys

from .listing import main

sys.exit(main())
