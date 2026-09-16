"""
broker.py — The heart of the pub/sub system: a thread-safe registry of
topics, services, and in-flight service calls, plus one method per
protocol operation.

--------------------------------------------------------------------------
BACKGROUND: threads, shared state, and why we need a lock at all
--------------------------------------------------------------------------
This gateway handles many TCP clients at once using one OS thread per
connection (see connection.py and server.py — this is the "thread-per-
connection" concurrency model). Each of those threads calls into this
single, shared `Broker` object: when client A calls `subscribe`, that runs
on A's thread; when client B calls `publish` a moment later, that runs
*concurrently* on B's thread. Both threads may be reading and writing the
same Python dict/set objects (`self._topics`, `self._services`, ...) at
literally the same instant.

Without protection, this is a classic **race condition**. As one concrete
example: Python's `dict`/`set` mutations are usually safe from silent
memory corruption thanks to the GIL (the Global Interpreter Lock, which
serializes *individual* bytecode operations), but a *sequence* of
operations — like "check if a topic exists, and if not, create it" — is
not atomic as a whole. Two threads could both see "topic doesn't exist
yet," both try to create it, and one thread's `TopicState` object gets
silently discarded when the second thread's assignment overwrites it,
along with whatever subscriber the first thread had just added. This
class of bug is intermittent, timing-dependent, and exactly the kind of
thing that only shows up under grading load, not in casual manual testing
— which is why we protect shared state with an explicit lock: it isn't
optional caution, it's a correctness requirement.

**The rule this file follows everywhere: hold `self._lock` only for
in-memory dict/set bookkeeping — a handful of Python operations, never
longer — and NEVER while doing socket I/O (`sink.send(...)`).**

Why does that second half matter so much? `sink.send()` (see
connection.py) is fast for a *healthy* connection, but a connection whose
peer has stopped reading can eventually make `send()`/its underlying
`socket.send()` block for a long time (the kernel's outbound TCP buffer
fills up and has nowhere to put more bytes). If we held the broker lock
while calling `sink.send()` to one slow subscriber, *every other thread*
in the whole gateway would stall waiting for that one lock — one
misbehaving client would freeze the entire system. The protocol doc says
plainly: "a slow client must not prevent unrelated clients from making
progress." Our answer to that requirement lives here: every method below
acquires the lock only to read/copy/mutate the registries, releases it,
and only *then* calls `sink.send()`. (`sink.send()` itself only enqueues
a message onto that connection's own outbound queue — see connection.py —
so even that call is fast; the actual, potentially-slow `socket.send()`
happens later, on that connection's dedicated writer thread, entirely
outside of any lock.)
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from protocol import publish_envelope, service_response, status

log = logging.getLogger("broker")

# How long the gateway waits for a provider to answer a forwarded
# call_service before giving up and synthesizing a result:false response.
# The protocol doc requires *some* bound ("Timeout duration is
# implementation-defined but must be bounded") without specifying a value.
SERVICE_CALL_TIMEOUT_S = 5.0


class MessageSink(Protocol):
    """Anything the broker can deliver a JSON message to.

    This is the abstraction that lets the broker treat a real TCP client
    (`Connection`, see connection.py) and an in-process service provider
    (`InProcessProvider`, see services/provider.py) identically: the
    broker only ever calls `.send(message)` on either kind of object and
    never needs to know or care whether that message ends up serialized
    onto a socket or handed directly to a Python function call. That
    uniformity is what lets `/plan_path` be registered later as "just
    another provider" with zero changes to this file.

    `id` is a short human-readable label (not the protocol's `id` field —
    unfortunately both concepts are called "id"; this one is used only in
    log messages to say *which* connection something happened to).
    """

    id: str

    def send(self, message: dict[str, Any]) -> None: ...


@dataclass(eq=False)
class TopicState:
    """Everything the broker tracks about one topic name.

    `eq=False` keeps the dataclass's identity-based default equality
    (`is`), which is what we want for dict/set membership here — two
    different `TopicState` objects should never compare equal just
    because their current subscriber sets happen to match.
    """

    subscribers: set[MessageSink] = field(default_factory=set)
    # `publishers` gates external publishes: `Broker.publish()` only
    # delivers a publish from an external connection (one that passes its
    # own `sink`) if that sink is currently a member of this set. A
    # connection joins it via `advertise` and leaves it via `unadvertise`
    # or disconnect (see `cleanup_connection`) — so unadvertising and then
    # immediately publishing again correctly reaches no one.
    publishers: set[MessageSink] = field(default_factory=set)


@dataclass(eq=False)
class PendingCall:
    """One in-flight `call_service` the broker is waiting on a provider to
    answer.

    Note what this is keyed by, and why: **the broker mints its own
    `provider_call_id`, distinct from the caller's own `id` field.** The
    protocol only guarantees a caller's id is unique *among that caller's
    own outstanding requests* — nothing stops two different connections
    from independently choosing the same id value (e.g. both picking
    `"call-1"`). If we stored pending calls in a dict keyed by the raw
    caller id, those two callers' entries would collide and one call could
    incorrectly complete the other's request. By minting a fresh, globally
    unique id here and remembering *which caller and which original id* it
    corresponds to, that collision becomes structurally impossible rather
    than something we have to detect and guard against.
    """

    provider_call_id: str
    service: str
    caller: MessageSink
    caller_id: Any
    provider: MessageSink
    timer: threading.Timer


class Broker:
    """The shared, thread-safe pub/sub + service registry.

    One `Broker` instance is created in `main.py` and shared by every
    `Connection` and every in-process service provider for the life of
    the process. Every public method here corresponds to one operation
    from spec/ROSBRIDGE_PROTOCOL.md.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._topics: dict[str, TopicState] = {}
        self._services: dict[str, MessageSink] = {}
        self._outstanding: dict[str, PendingCall] = {}
        # itertools.count gives an infinite, thread-safe-enough (we only
        # ever call next() while already holding self._lock) source of
        # increasing integers, used to build unique provider-side call ids.
        self._call_counter = itertools.count(1)

    # -- Topics --------------------------------------------------------

    def advertise(self, sink: MessageSink, topic: str) -> None:
        with self._lock:
            self._topics.setdefault(topic, TopicState()).publishers.add(sink)

    def unadvertise(self, sink: MessageSink, topic: str) -> None:
        with self._lock:
            state = self._topics.get(topic)
            if state is not None:
                state.publishers.discard(sink)

    def subscribe(self, sink: MessageSink, topic: str, request_id: Any = None) -> None:
        with self._lock:
            self._topics.setdefault(topic, TopicState()).subscribers.add(sink)
        # Sent *after* releasing the lock, per the module docstring's rule.
        sink.send(status("info", f"subscribed to {topic}", request_id))

    def unsubscribe(self, sink: MessageSink, topic: str) -> None:
        with self._lock:
            state = self._topics.get(topic)
            if state is not None:
                state.subscribers.discard(sink)

    def publish(self, topic: str, msg: Any, sink: Optional[MessageSink] = None) -> None:
        """Deliver `msg` to every current subscriber of `topic`.

        `sink` identifies who's publishing, when known. External clients
        always pass it (see connection.py's `_handle_publish`): a
        publish is only delivered if `sink` is *currently* advertised as
        a publisher of this exact topic — calling `unadvertise` (or
        disconnecting) and then publishing again must not still reach
        subscribers. `sink` is optional, not required, because internal
        runtime components (e.g. planner.py publishing `/path` after a
        successful plan) aren't connections and have no advertise
        registration to check; passing nothing preserves the old
        unconditional behavior for exactly that kind of trusted caller.
        """
        with self._lock:
            state = self._topics.get(topic)
            if sink is not None and (state is None or sink not in state.publishers):
                return
            # Copy the subscriber set while we hold the lock, so we can
            # iterate and call .send() on each one *after* releasing it.
            # Iterating the live set without a lock held would itself be a
            # race (another thread could add/remove a subscriber mid-loop).
            subscribers = set(state.subscribers) if state is not None else set()
        envelope = publish_envelope(topic, msg)
        for subscriber in subscribers:
            subscriber.send(envelope)

    # -- Services --------------------------------------------------------

    def advertise_service(self, sink: MessageSink, service: str, request_id: Any = None) -> None:
        # Policy: first-registrant-wins. If a connection already provides
        # this service, a second connection's attempt to advertise the
        # same name is rejected outright rather than silently replacing
        # the first provider. This keeps "who currently owns this
        # service" unambiguous at every moment — useful both for grading
        # determinism and for a student debugging why a call went to an
        # unexpected place. A provider that legitimately wants to give up
        # the name must call `unadvertise_service` (or disconnect, which
        # frees it automatically via cleanup_connection below).
        with self._lock:
            current = self._services.get(service)
            already_owns_it = current is sink
            taken_by_someone_else = current is not None and not already_owns_it
            if not taken_by_someone_else:
                self._services[service] = sink
        if taken_by_someone_else:
            sink.send(status("error", f"service {service} already has a provider", request_id))
        else:
            sink.send(status("info", f"registered {service}", request_id))

    def unadvertise_service(self, sink: MessageSink, service: str) -> None:
        with self._lock:
            # Ownership check: only the *current* provider may withdraw.
            # This stops a stray/late unadvertise_service from some other
            # connection accidentally freeing a service it never owned.
            if self._services.get(service) is sink:
                del self._services[service]

    def call_service(self, caller: MessageSink, caller_id: Any, service: str, args: Any) -> None:
        with self._lock:
            provider = self._services.get(service)
            if provider is None:
                has_provider = False
            else:
                has_provider = True
                provider_call_id = f"svc-{next(self._call_counter)}"
                timer = threading.Timer(
                    SERVICE_CALL_TIMEOUT_S, self._on_timeout, args=(provider_call_id,)
                )
                # A daemon thread never blocks Python's interpreter from
                # exiting; it's a safety net alongside the orderly shutdown
                # sequence in main.py (see that file's docstring).
                timer.daemon = True
                self._outstanding[provider_call_id] = PendingCall(
                    provider_call_id=provider_call_id,
                    service=service,
                    caller=caller,
                    caller_id=caller_id,
                    provider=provider,
                    timer=timer,
                )
        if not has_provider:
            # A missing provider is known *synchronously* — there is
            # nothing to wait for, so we respond immediately rather than
            # starting a timer just to have it fire uselessly later.
            caller.send(service_response(service, caller_id, {}, False, "no provider"))
            return
        # Only start the timer, and only forward the call to the provider,
        # after releasing the lock: both are actions with externally
        # observable side effects (a running timer, an outbound message),
        # and per this module's rule, side effects happen outside the lock.
        timer.start()
        provider.send(
            {"op": "call_service", "service": service, "id": provider_call_id, "args": args}
        )

    def service_response(
        self, sender: MessageSink, provider_call_id: Any, values: Any, result: bool, status_text: str
    ) -> None:
        """Handle a `service_response` arriving from a provider, and
        forward it to whichever caller is actually waiting on it.
        """
        with self._lock:
            pending = self._outstanding.get(provider_call_id)
            # A response is only valid if (a) we still have a pending call
            # under this exact id — it may have already been answered, or
            # timed out, or the id may simply be bogus — AND (b) it came
            # from the connection we actually forwarded *that* call to.
            # Requirement (b) is what stops a random or misbehaving
            # connection from completing someone else's call just by
            # guessing or replaying an id.
            valid = pending is not None and pending.provider is sender
            if valid:
                del self._outstanding[provider_call_id]
                pending.timer.cancel()
        if not valid:
            log.warning("ignoring service_response for unknown/foreign call id %r", provider_call_id)
            return
        assert pending is not None  # for type-checkers; `valid` already guarantees this
        pending.caller.send(
            service_response(pending.service, pending.caller_id, values, result, status_text)
        )

    def _on_timeout(self, provider_call_id: str) -> None:
        """Runs on a `threading.Timer`'s own background thread when a
        provider fails to answer within SERVICE_CALL_TIMEOUT_S.

        There's an inherent race here worth naming explicitly: the real
        `service_response` could arrive at almost the same instant the
        timer fires. We resolve it with `dict.pop(..., None)`: whichever
        of "the real response" and "the timeout" acquires the lock first
        removes the entry and wins; the other finds it already gone and
        does nothing. Neither side needs to know about the other for this
        to be correct — the shared dict is the single source of truth.
        """
        with self._lock:
            pending = self._outstanding.pop(provider_call_id, None)
        if pending is not None:
            pending.caller.send(
                service_response(pending.service, pending.caller_id, {}, False, "timeout")
            )

    # -- Connection lifecycle --------------------------------------------

    def cleanup_connection(self, sink: MessageSink) -> None:
        """Called exactly once when a connection's socket closes (see
        connection.py), to remove every trace of it from the broker.

        Implementation choice worth calling out: rather than having each
        `Connection` separately track "which topics/services/calls do I
        own" and hand that list back to the broker on disconnect, this
        method simply scans the broker's own registries and discards the
        departing `sink` everywhere it appears. That trades a little bit
        of extra work per disconnect (proportional to the number of
        topics/services/outstanding calls that currently exist) for a
        real simplicity win: there is exactly one place — this method —
        that decides what "this connection's stuff" means, instead of two
        copies of that bookkeeping (one in `Connection`, one in `Broker`)
        that could drift out of sync with each other. For a course-scale
        gateway (a handful of topics/services/clients, not thousands),
        that trade is clearly worth it.
        """
        orphaned_as_provider: list[PendingCall] = []
        with self._lock:
            for state in self._topics.values():
                state.subscribers.discard(sink)
                state.publishers.discard(sink)

            dead_services = [name for name, provider in self._services.items() if provider is sink]
            for name in dead_services:
                del self._services[name]

            for call_id, pending in list(self._outstanding.items()):
                if pending.provider is sink:
                    # This connection was providing an answer someone else
                    # is still waiting on; it can never answer now.
                    del self._outstanding[call_id]
                    pending.timer.cancel()
                    orphaned_as_provider.append(pending)
                elif pending.caller is sink:
                    # This connection was waiting on an answer, but it's
                    # gone now, so there is no one left to notify.
                    del self._outstanding[call_id]
                    pending.timer.cancel()
        for pending in orphaned_as_provider:
            pending.caller.send(
                service_response(pending.service, pending.caller_id, {}, False, "provider disconnected")
            )
