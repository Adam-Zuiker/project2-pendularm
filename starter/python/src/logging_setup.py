"""
logging_setup.py — Configure Python's standard `logging` module so all
diagnostic output goes to **stderr**, never stdout.

Why this matters, specifically for this project: a black-box grader
(Autograder.io, per spec/PROJECT1_ASTAR.md) interacts with this program
*only* over the TCP/JSON protocol on 127.0.0.1:9095 — it never reads this
process's stdout/stderr as part of grading. Even so, the project spec is
explicit: "Keep diagnostic output separate from the TCP/JSON protocol
stream." Routing every log message through the standard `logging` module
configured to write to stderr, instead of scattering ad hoc `print()`
calls (which default to stdout), makes that separation structural — true
by construction in every file — rather than a convention we have to
remember to follow correctly everywhere by hand.
"""

from __future__ import annotations

import logging
import sys


def configure(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
