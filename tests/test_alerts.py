from datetime import datetime
from pathlib import Path

from src import alerts


def make_subscription(path: Path) -> dict:
    created = alerts.subscribe(path, "person@example.com", "delhi", "HIGH", "en")
    assert alerts.confirm(path, created["token"])
    return alerts.confirmed_for_city(path, "delhi")[0]


def test_double_opt_in_and_unsubscribe(tmp_path: Path):
    path = tmp_path / "alerts.sqlite3"
    created = alerts.subscribe(path, "person@example.com", "delhi", "HIGH", "en")
    assert alerts.confirmed_for_city(path, "delhi") == []
    assert alerts.confirm(path, created["token"])
    assert len(alerts.confirmed_for_city(path, "delhi")) == 1
    assert alerts.unsubscribe(path, created["token"])
    assert alerts.confirmed_for_city(path, "delhi") == []


def test_rendering_has_every_language_and_level():
    for language in ("en", "hi", "mr"):
        for level in alerts.RISK_RANK:
            message = alerts.render_alert(
                "Delhi", "2026-05-01", level, 0.8, language, "https://example.test/unsub"
            )
            assert message["subject"]
            assert "https://example.test/unsub" in message["body"]


def test_delivery_deduplicates_and_respects_quiet_hours(tmp_path: Path):
    path, sent = tmp_path / "alerts.sqlite3", []
    subscription = make_subscription(path)
    midday = datetime(2026, 5, 1, 12, tzinfo=alerts.IST)

    def sender(email, message):
        sent.append((email, message))

    assert alerts.deliver_if_due(path, subscription, "2026-05-02", "HIGH", 0.8, midday, sender)
    assert not alerts.deliver_if_due(path, subscription, "2026-05-02", "HIGH", 0.8, midday, sender)
    assert not alerts.deliver_if_due(
        path,
        subscription,
        "2026-05-03",
        "HIGH",
        0.8,
        datetime(2026, 5, 1, 23, tzinfo=alerts.IST),
        sender,
    )
    assert len(sent) == 1


def test_delivery_failure_does_not_record_success(tmp_path: Path):
    path = tmp_path / "alerts.sqlite3"
    subscription = make_subscription(path)
    assert not alerts.deliver_if_due(
        path,
        subscription,
        "2026-05-02",
        "EXTREME",
        0.9,
        datetime(2026, 5, 1, 12, tzinfo=alerts.IST),
        lambda *_: (_ for _ in ()).throw(RuntimeError("mock failure")),
    )
