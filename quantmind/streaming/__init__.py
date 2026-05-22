"""Real-time streaming of regime updates.

A pluggable message bus (in-memory by default, real Kafka when a broker is
configured) plus a regime update stream consumed over a WebSocket. The in-memory
bus means the live dashboard works with no external broker; set
``KAFKA_BOOTSTRAP_SERVERS`` and use :class:`KafkaBus` to publish to Kafka.
"""

from quantmind.streaming.bus import InMemoryBus, KafkaBus
from quantmind.streaming.stream import iter_regime_updates, regime_stream

__all__ = ["InMemoryBus", "KafkaBus", "iter_regime_updates", "regime_stream"]
