import asyncio

from fastapi.testclient import TestClient

from quantmind.api.main import app
from quantmind.data import generate_regime_series
from quantmind.streaming import InMemoryBus, iter_regime_updates

client = TestClient(app)


def test_in_memory_bus_pubsub():
    async def run():
        bus = InMemoryBus()
        q = bus.subscribe()
        assert bus.n_subscribers == 1
        await bus.publish({"label": "Bull"})
        msg = await asyncio.wait_for(q.get(), timeout=1.0)
        assert msg == {"label": "Bull"}
        bus.unsubscribe(q)
        assert bus.n_subscribers == 0

    asyncio.run(run())


def test_iter_regime_updates_payload():
    updates = iter_regime_updates(generate_regime_series(n_days=400, seed=3)["close"])
    assert len(updates) > 100
    assert set(updates[0]) == {"type", "date", "close", "label", "confidence"}
    assert updates[0]["label"] in {"Bull", "Bear", "Neutral"}


def test_websocket_streams_ticks_then_completes():
    with client.websocket_connect("/ws/regime?source=synthetic&speed=0&limit=8") as ws:
        ticks = [ws.receive_json() for _ in range(8)]
        assert all(t["type"] == "tick" for t in ticks)
        assert set(ticks[0]) == {"type", "date", "close", "label", "confidence"}
        assert ws.receive_json()["type"] == "complete"
