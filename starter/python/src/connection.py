"""
connection.py — Everything associated with one connected TCP client: a
socket, a reader thread, a writer thread, and the dispatch logic that
turns parsed protocol messages into calls on the shared `Broker`.

--------------------------------------------------------------------------
BACKGROUND: why *two* threads (reader + writer) per connection, not one?
--------------------------------------------------------------------------
A socket supports two independent operations that can each block the
calling thread for an unpredictable amount of time:

  - `recv()` blocks until the peer sends more bytes (or closes).
  - `send()` blocks if the kernel's outbound buffer for this socket is
    full — which happens if the peer is slow to read, or has stopped
    reading altogether.

If one thread tried to do both jobs — read a request, maybe write a
response, loop — a slow-to-read peer could leave that thread stuck inside
`send()` for a long time, unable to get back around to `recv()` and
notice, say, that the same peer also wants to send us a new message. Two
different pieces of "waiting" would be tangled into one thread with no
easy way to make progress on either independently.

Splitting the two directions into separate threads sidesteps this
completely, and lets each thread be written as a *simple, fully blocking
loop* with no timeouts or polling needed:

  - The **reader thread** does nothing but `recv()` → parse → dispatch,
    forever, until the peer disconnects.
  - The **writer thread** does nothing but pull the next outbound message
    off this connection's own `queue.Queue` → `send()` it, forever, until
    told to stop.

Other parts of the system (the broker, when relaying a `publish` or a
`service_response` to this connection) never call `socket.send()`
directly. They call this connection's `send(message)` method, which only
does `self._outbound.put(message)` — appending to a thread-safe queue is
fast and never blocks the caller, no matter how slow *this* connection's
peer currently is at reading. The actual, potentially-slow `socket.send()`
call happens later, on this connection's own writer thread, completely
decoupled from whatever thread produced the message. This is the other
half of the "never do I/O while holding the broker's lock, and never let
one slow client block another" story that broker.py's module docstring
introduces — read that one first if you haven't.
"""

from __future__ import annotations

import itertools
import logging
import queue
import socket
import threading
from typing import Any, Callable, Dict, Optional

from broker import Broker
from protocol import LineReader, ProtocolError, encode, status

log = logging.getLogger("connection")

# A unique sentinel object (not any real protocol message) placed on the
# outbound queue to tell the writer thread "stop, there is nothing more
# to send, exit your loop." Using a dedicated `object()` instead of, say,
# `None` avoids any ambiguity with a legitimate (if unusual) message.
_SHUTDOWN = object()


class Connection:
    """One connected TCP client, and everything needed to serve it.

    Implements the `MessageSink` protocol from broker.py: it has an `id`
    and a `send(message)` method, which is all the broker ever needs to
    know about it.
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        sock: socket.socket,
        addr: tuple[str, int],
        broker: Broker,
        on_close: Optional[Callable[["Connection"], None]] = None,
    ) -> None:
        self.id = f"conn-{next(Connection._id_counter)}@{addr[0]}:{addr[1]}"
        self._sock = sock
        self._broker = broker
        # Optional callback so server.py can drop this Connection from its
        # own bookkeeping set once it's done, instead of accumulating
        # references to every connection that ever existed for the life
        # of the process.
        self._on_close = on_close
        # `queue.Queue` is already internally thread-safe (it has its own
        # lock), so multiple threads (many different broker callers) can
        # safely call `.put()` on it concurrently without us adding any
        # locking of our own here.
        self._outbound: "queue.Queue[Any]" = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_loop, name=f"{self.id}-reader", daemon=True
        )
        self._writer_thread = threading.Thread(
            target=self._write_loop, name=f"{self.id}-writer", daemon=True
        )
        # Guards against running shutdown/cleanup twice: both the reader
        # thread (on recv() failure/EOF) and an external caller (server.py
        # tearing everything down) can independently decide "this
        # connection needs to close," and we must only tell the broker to
        # clean it up exactly once.
        self._close_lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        self._reader_thread.start()
        self._writer_thread.start()

    # -- MessageSink interface -------------------------------------------

    def send(self, message: Dict[str, Any]) -> None:
        """Queue `message` for delivery to this client. Safe to call from
        any thread; never blocks. See the module docstring for why."""
        self._outbound.put(message)

    # -- Reader thread -----------------------------------------------------

    def _read_loop(self) -> None:
        reader = LineReader()
        try:
            while True:
                # recv(4096): ask for up to 4096 bytes. TCP makes no
                # promise you'll get that many, or even one full message
                # — you might get 3 bytes, or 3 messages, in a single
                # call. That's exactly why LineReader exists (see
                # protocol.py): it buffers across calls and hands back
                # only whole, parsed messages.
                chunk = self._sock.recv(4096)
                if not chunk:
                    # recv() returning b"" (not raising) is how Python
                    # sockets signal a clean, orderly peer shutdown ("I'm
                    # done sending, but see TCP's half-close semantics for
                    # why this isn't necessarily the whole connection").
                    # For our purposes: no more requests are coming.
                    break
                try:
                    messages = list(reader.feed(chunk))
                except ProtocolError as exc:
                    # Malformed bytes on the wire. The protocol doc allows
                    # us to reject the request or close the connection;
                    # we choose the more forgiving option — tell the
                    # client what went wrong and keep listening — since a
                    # single bad line elsewhere in a test script shouldn't
                    # need to tear down an otherwise-fine connection.
                    self.send(status("error", str(exc)))
                    continue
                for message in messages:
                    self._dispatch(message)
        except OSError:
            # The socket was closed or reset out from under us — normal
            # and expected both for an abrupt client disconnect and for
            # our own shutdown path calling close() (see below), which
            # deliberately shuts the socket down to unblock this exact
            # recv() call.
            pass
        finally:
            self._shutdown()

    def _dispatch(self, message: Dict[str, Any]) -> None:
        op = message.get("op")
        request_id = message.get("id")
        handler = self._HANDLERS.get(op)
        if handler is None:
            self.send(status("error", f"unknown op {op!r}", request_id))
            return
        try:
            handler(self, message)
        except (KeyError, TypeError, ValueError) as exc:
            # A required field was missing or the wrong type. Caught here,
            # per-message, so one malformed request can't crash this
            # connection's reader thread — and per the protocol doc, it
            # must never crash the gateway or any *other* connection
            # either, which is guaranteed structurally: every connection
            # has its own reader thread with its own try/except.
            log.info("malformed %r request on %s: %s", op, self.id, exc)
            self.send(status("error", f"malformed {op} request: {exc}", request_id))

    # -- One handler method per protocol operation -------------------------
    # Each of these does the minimum: pull the required fields out of the
    # raw dict (a missing key raises KeyError, caught by _dispatch above)
    # and call straight into the broker. All the actual pub/sub and
    # service logic lives in broker.py — this file is purely "wire format
    # in, broker call out."

    def _handle_advertise(self, message: Dict[str, Any]) -> None:
        self._broker.advertise(self, message["topic"])

    def _handle_unadvertise(self, message: Dict[str, Any]) -> None:
        self._broker.unadvertise(self, message["topic"])

    def _handle_publish(self, message: Dict[str, Any]) -> None:
        # Passing `self` is what makes the broker enforce "must be
        # currently advertised to publish" for external clients — see
        # Broker.publish()'s docstring for why internal callers (like
        # planner.py publishing /path) omit it instead.
        self._broker.publish(message["topic"], message["msg"], self)

    def _handle_subscribe(self, message: Dict[str, Any]) -> None:
        self._broker.subscribe(self, message["topic"], message.get("id"))

    def _handle_unsubscribe(self, message: Dict[str, Any]) -> None:
        self._broker.unsubscribe(self, message["topic"])

    def _handle_advertise_service(self, message: Dict[str, Any]) -> None:
        self._broker.advertise_service(self, message["service"], message.get("id"))

    def _handle_unadvertise_service(self, message: Dict[str, Any]) -> None:
        self._broker.unadvertise_service(self, message["service"])

    def _handle_call_service(self, message: Dict[str, Any]) -> None:
        # `id` is required for call_service specifically (the protocol
        # doc says so explicitly) — a missing id is therefore just as
        # malformed as a missing `service`, and KeyError handles both the
        # same way via _dispatch's except clause.
        self._broker.call_service(
            self, message["id"], message["service"], message.get("args")
        )

    def _handle_service_response(self, message: Dict[str, Any]) -> None:
        self._broker.service_response(
            self,
            message["id"],
            message.get("values"),
            message["result"],
            message.get("status", ""),
        )

    # Built after the handler methods above so it can reference them by
    # name; this is an ordinary class attribute (a dict of unbound
    # functions), looked up and called as `handler(self, message)` in
    # _dispatch. Using a dict instead of a chain of `if op == ...: ...`
    # makes "unknown op" a simple, correct-by-construction fallthrough
    # (`.get(op)` returns None) rather than something we could forget an
    # `elif` branch for.
    _HANDLERS: Dict[str, Callable[["Connection", Dict[str, Any]], None]] = {
        "advertise": _handle_advertise,
        "unadvertise": _handle_unadvertise,
        "publish": _handle_publish,
        "subscribe": _handle_subscribe,
        "unsubscribe": _handle_unsubscribe,
        "advertise_service": _handle_advertise_service,
        "unadvertise_service": _handle_unadvertise_service,
        "call_service": _handle_call_service,
        "service_response": _handle_service_response,
    }

    # -- Writer thread -----------------------------------------------------

    def _write_loop(self) -> None:
        try:
            while True:
                message = self._outbound.get()  # blocks until something is queued
                if message is _SHUTDOWN:
                    break
                try:
                    self._sock.sendall(encode(message))
                except OSError:
                    # The peer is gone or the socket is closing; nothing
                    # further can be sent, so this thread's job is done.
                    break
        finally:
            try:
                self._sock.close()
            except OSError:
                pass

    # -- Shutdown ------------------------------------------------------

    def close(self) -> None:
        """External trigger to disconnect this client (used by server.py
        during process-wide shutdown). Safe to call even if the
        connection is already closing on its own."""
        try:
            # shutdown(SHUT_RDWR) asks the OS to stop allowing reads and
            # writes on this socket. Its main effect here: it forces the
            # reader thread's blocked recv() call to return immediately
            # (with either b"" or an OSError, both handled above) instead
            # of waiting for a peer that may never send anything else.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already shut down/closed — fine, close() must be idempotent
        self._shutdown()

    def _shutdown(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        # Tell the broker to forget this connection existed: drop it from
        # every topic's subscriber/publisher set, release any service it
        # was providing, and resolve any outstanding calls it was part of.
        # See broker.py's cleanup_connection docstring for the full story.
        self._broker.cleanup_connection(self)
        if self._on_close is not None:
            self._on_close(self)
        # Wake the writer thread up so it can exit its loop instead of
        # blocking forever on an outbound queue nothing will ever add to
        # again.
        self._outbound.put(_SHUTDOWN)
