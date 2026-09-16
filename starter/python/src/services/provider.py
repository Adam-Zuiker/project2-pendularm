"""
services/provider.py — Adapts a plain Python function into a service
provider the broker can call, without ever touching a real socket.

--------------------------------------------------------------------------
BACKGROUND: why in-process, when a "real" provider is a TCP client?
--------------------------------------------------------------------------
From the *external* protocol's point of view, a service provider is just
another TCP client: it connects, sends `advertise_service`, and later
receives forwarded `call_service` requests and answers them with
`service_response`. Nothing in the spec requires our own built-in
`/heapify` and `/heap_sort` services to *literally* be separate TCP
clients dialing back into our own listening socket — the project brief
says outright that "the internal design is entirely yours... There is no
required internal wire protocol." A loopback socket would work, but it
buys us nothing here and costs real complexity: our own runtime would
need to connect to *itself* over the network, which means the listener
has to be up and accepting connections *before* we can register our own
built-in services — an awkward startup-ordering dependency for no
external benefit.

Instead, `InProcessProvider` implements the exact same `MessageSink`
interface (`id` + `send(message)`) that a real socket `Connection` does
(see broker.py and connection.py), but its `send()` doesn't serialize
anything to a wire — it just hands the forwarded call off to a plain
Python function on a dedicated worker thread, and reports the result
straight back into the broker via the *same* `Broker.service_response()`
method a real socket-based provider's `Connection` would call after
parsing a `service_response` line off the wire. The broker itself cannot
tell the difference between an in-process provider and a real external
one — which is exactly the point: it means a *future* provider (like the
A* planner's `/plan_path`) can be swapped from "a plain function" to "a
real subprocess talking the wire protocol" later without changing
`broker.py` at all.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Tuple

from broker import Broker

log = logging.getLogger("services.provider")

# The type of a service handler: takes the parsed `args` value from a
# call_service request, and returns (values, result, status_text) — the
# same three pieces of information a real provider would put into its
# outbound service_response envelope.
Handler = Callable[[Any], Tuple[Any, bool, str]]

_SHUTDOWN = object()


class InProcessProvider:
    """A `MessageSink` that answers calls by running a Python function on
    its own worker thread, instead of by talking to a socket."""

    def __init__(self, service: str, handler: Handler) -> None:
        self.id = f"provider:{service}"
        self._service = service
        self._handler = handler
        self._inbound: "queue.Queue[Any]" = queue.Queue()
        self._broker: Broker | None = None  # set by register(), below
        self._thread = threading.Thread(
            target=self._run, name=f"provider-{service}", daemon=True
        )

    def start(self, broker: Broker) -> None:
        self._broker = broker
        self._thread.start()

    def stop(self) -> None:
        """Ask the worker thread to exit and wait (briefly) for it to.
        Called from main.py during shutdown."""
        self._inbound.put(_SHUTDOWN)
        self._thread.join(timeout=2.0)

    # -- MessageSink interface -------------------------------------------

    def send(self, message: dict) -> None:
        """Called by the broker. Two kinds of messages arrive here:

        - A forwarded `call_service` (sent by `Broker.call_service` right
          after we're chosen as the current provider): queue it for the
          worker thread to actually run.
        - A `status` message (sent by `Broker.advertise_service` to
          confirm — or reject — our own registration attempt, made in
          `register()` below): there is no human or client reading these
          for an in-process provider, so we just log it. An "error" level
          status here is worth noticing during development — it would
          mean, e.g., something else already registered this exact
          service name before we did.
        """
        op = message.get("op")
        if op == "call_service":
            self._inbound.put(message)
        elif op == "status":
            level = logging.INFO if message.get("level") != "error" else logging.WARNING
            log.log(level, "%s registration status: %s", self.id, message.get("msg"))
        else:
            log.warning("%s received unexpected message: %r", self.id, message)

    # -- Worker thread -----------------------------------------------------

    def _run(self) -> None:
        assert self._broker is not None, "start() must be called before the worker runs"
        while True:
            message = self._inbound.get()
            if message is _SHUTDOWN:
                return
            provider_call_id = message.get("id")
            args = message.get("args")
            try:
                values, result, status_text = self._handler(args)
            except Exception as exc:  # noqa: BLE001 - a handler bug must not
                # permanently kill this service for every future caller;
                # report it as an ordinary failed call instead of letting
                # the exception propagate and silently end this thread.
                log.exception("%s handler raised for args=%r", self.id, args)
                values, result, status_text = {}, False, f"internal error: {exc}"
            self._broker.service_response(self, provider_call_id, values, result, status_text)


def register(broker: Broker, service: str, handler: Handler) -> InProcessProvider:
    """Create, start, and register an in-process provider for `service`.

    This is the one function later phases (the A* planner registering
    `/plan_path`) will call too — see services/planner_stub.py.
    """
    provider = InProcessProvider(service, handler)
    provider.start(broker)
    broker.advertise_service(provider, service)
    return provider
