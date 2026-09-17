"""main.py — Entry point for the Project 2 Pendularm runtime. This is what
`make run` executes: launch in the foreground, expose 127.0.0.1:9095, read
`ARM_SIM_LINKS` (`"2"` or `"3"`, default `"2"`) to pick the link count, fail
loudly if startup fails, and shut down cleanly on SIGINT/SIGTERM.

Structurally identical to Project 1's main.py (see that file's docstring
for the full explanation of the signal-handling dance around a blocked
`accept()` call) — only the set of registered services differs.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

import arm_sim_node
import ik_action_node
import ik_node
import ik_trial_node
import logging_setup
from broker import Broker
from server import TCPServer

log = logging.getLogger("main")


def _arm_sim_links() -> int:
    raw = os.environ.get("ARM_SIM_LINKS", "2")
    if raw not in ("2", "3"):
        log.warning("ARM_SIM_LINKS=%r is not \"2\" or \"3\"; defaulting to 2", raw)
        return 2
    return int(raw)


def main() -> int:
    logging_setup.configure()

    n_links = _arm_sim_links()
    log.info("configured for a %d-link arm", n_links)

    broker = Broker()
    stoppables, runtime = arm_sim_node.register(broker, n_links)
    stoppables += ik_node.register(broker, runtime)
    ik_action_providers, ik_action_manager = ik_action_node.register(broker, runtime)
    stoppables += ik_action_providers
    ik_trial_stoppables, _ik_trial = ik_trial_node.register(broker, runtime, ik_action_manager)
    stoppables += ik_trial_stoppables
    log.info("registered: %s", ", ".join(s.id for s in stoppables))

    server = TCPServer(broker)
    try:
        server.bind_and_listen()
    except OSError as exc:
        log.error("could not bind 127.0.0.1:9095: %s", exc)
        return 1

    def handle_signal(signum: int, frame) -> None:  # noqa: ANN001 - stdlib signature
        log.info("received signal %s, shutting down", signal.Signals(signum).name)
        server.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info("Project 2 runtime ready on 127.0.0.1:9095")
    try:
        server.serve_forever()
    finally:
        for stoppable in stoppables:
            stoppable.stop()
    log.info("shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
