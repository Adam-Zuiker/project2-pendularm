"""
server.py — The TCP listener: binds `127.0.0.1:9095`, accepts incoming
connections, and hands each one off to its own `Connection`.

--------------------------------------------------------------------------
BACKGROUND: listening sockets, `accept()`, and why the accept loop is
simple on purpose
--------------------------------------------------------------------------
A "listening" socket is different from a normal connected socket: you
never `send()`/`recv()` on it directly. Instead, the sequence is:

    1. `socket.socket(...)`  — create a socket object.
    2. `.bind((host, port))` — claim a specific address/port so clients
       can find us. `127.0.0.1` is the IPv4 loopback address: it only
       accepts connections *from this same machine*, never over a real
       network. `9095` is just the port number the assignment fixes.
    3. `.listen()`           — tell the OS "start queuing incoming
       connection attempts to this address/port; I'll come get them."
       Before this call, a client trying to connect would simply be
       refused (nothing is listening yet).
    4. `.accept()`           — pull the next completed incoming connection
       off that queue. This call *blocks* until a client connects, and
       when it returns, it hands back a **new, separate socket object**
       already connected to that one specific client — the original
       listening socket keeps listening for the *next* client and is
       never used to exchange actual data.

This module's whole job is steps 1–3 once at startup, then repeatedly
calling `accept()` in a loop and spinning up a `Connection` (its own
reader + writer threads, see connection.py) for each socket it gets back.
The accept loop itself does no protocol work at all — that separation is
deliberate, so "how do we accept new clients" and "how do we talk to an
already-connected client" stay two independent, easy-to-reason-about
pieces of code.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Optional

from broker import Broker
from connection import Connection

log = logging.getLogger("server")

HOST = "127.0.0.1"
PORT = 9095


class TCPServer:
    """Owns the listening socket and the set of currently-live connections."""

    def __init__(self, broker: Broker, host: str = HOST, port: int = PORT) -> None:
        self._broker = broker
        self._host = host
        self._port = port
        self._listen_sock: Optional[socket.socket] = None
        self._connections_lock = threading.Lock()
        self._connections: set[Connection] = set()
        self._shutting_down = False

    def bind_and_listen(self) -> None:
        """Claim the address/port and start queuing incoming connections.

        Raises `OSError` (e.g. "Address already in use" if something else
        already bound this port) if this fails — left uncaught so
        `main.py` can let it propagate and exit nonzero, per the project
        spec's requirement that `make run` "fails if required startup
        fails" rather than silently limping along without a working port.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR lets us re-bind this exact host/port immediately
        # after a previous run of this same program exited, instead of
        # the OS holding the port in a brief unusable "TIME_WAIT" state.
        # Extremely common on any TCP server you restart during
        # development; without it, running `make run` twice in a row
        # (stop, then immediately start again) can spuriously fail with
        # "Address already in use" even though nothing is actually using
        # it anymore.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen()
        self._listen_sock = sock
        log.info("listening on %s:%d", self._host, self._port)

    def serve_forever(self) -> None:
        """Run the accept loop on the calling thread. Blocks until
        `shutdown()` is called from another thread (e.g. a signal
        handler in main.py), at which point this returns normally.
        """
        assert self._listen_sock is not None, "call bind_and_listen() first"
        while True:
            try:
                client_sock, addr = self._listen_sock.accept()
            except OSError:
                # Raised when shutdown() closes the listening socket out
                # from under this blocked accept() call — our signal to
                # stop looping, not an error to report.
                if self._shutting_down:
                    return
                raise
            connection = Connection(
                client_sock, addr, self._broker, on_close=self._forget_connection
            )
            with self._connections_lock:
                self._connections.add(connection)
            connection.start()
            log.info("accepted %s", connection.id)

    def _forget_connection(self, connection: Connection) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def shutdown(self) -> None:
        """Stop accepting new connections and disconnect every current
        client. Safe to call from a different thread than the one running
        `serve_forever()` — that's the whole point: a signal handler
        calls this from the main thread while `serve_forever()`'s blocked
        `accept()` call is sitting on some other thread, or vice versa
        depending on how main.py is structured (see that file).
        """
        self._shutting_down = True
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
        with self._connections_lock:
            live_connections = list(self._connections)
        for connection in live_connections:
            connection.close()
