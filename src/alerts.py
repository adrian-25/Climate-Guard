"""Opt-in alert subscriptions, rendering, and delivery seams for ClimateGuard."""

import json
import logging
import os
import re
import secrets
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

RISK_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EXTREME": 3}
IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("climateguard.alerts")


def database_path(root: Path) -> Path:
    return Path(os.getenv("ALERT_DATABASE_PATH", str(root / "runtime" / "climateguard.sqlite3")))


def initialise(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS alert_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, city TEXT NOT NULL,
            minimum_risk TEXT NOT NULL, language TEXT NOT NULL, token TEXT UNIQUE NOT NULL,
            confirmed INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, confirmed_at TEXT
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS alert_deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT, subscription_id INTEGER NOT NULL,
            forecast_date TEXT NOT NULL, risk_level TEXT NOT NULL, status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 1, error TEXT, created_at TEXT NOT NULL,
            UNIQUE(subscription_id, forecast_date, risk_level, status)
        )""")


def valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value))


def subscribe(path: Path, email: str, city: str, minimum_risk: str, language: str) -> dict:
    initialise(path)
    if not valid_email(email):
        raise ValueError("Enter a valid email address.")
    if minimum_risk not in RISK_RANK or language not in {"en", "hi", "mr"}:
        raise ValueError("Unsupported alert preference.")
    token = secrets.token_urlsafe(32)
    now = datetime.now(IST).isoformat()
    with sqlite3.connect(path) as connection:
        existing = connection.execute(
            "SELECT id, token FROM alert_subscriptions WHERE email=? AND city=? AND active=1",
            (email.lower(), city),
        ).fetchone()
        if existing:
            connection.execute(
                "UPDATE alert_subscriptions SET minimum_risk=?, language=? WHERE id=?",
                (minimum_risk, language, existing[0]),
            )
            return {"id": existing[0], "token": existing[1], "existing": True}
        cursor = connection.execute(
            "INSERT INTO alert_subscriptions(email, city, minimum_risk, language, token, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (email.lower(), city, minimum_risk, language, token, now),
        )
    return {"id": cursor.lastrowid, "token": token, "existing": False}


def confirm(path: Path, token: str) -> bool:
    initialise(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            "UPDATE alert_subscriptions SET confirmed=1, confirmed_at=? WHERE token=? AND active=1",
            (datetime.now(IST).isoformat(), token),
        )
    return bool(cursor.rowcount)


def unsubscribe(path: Path, token: str) -> bool:
    initialise(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            "UPDATE alert_subscriptions SET active=0 WHERE token=?", (token,)
        )
    return bool(cursor.rowcount)


def confirmed_for_city(path: Path, city: str) -> list[dict]:
    initialise(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT id, email, city, minimum_risk, language, token FROM alert_subscriptions WHERE city=? AND confirmed=1 AND active=1",
            (city,),
        ).fetchall()
    return [
        dict(zip(("id", "email", "city", "minimum_risk", "language", "token"), row)) for row in rows
    ]


def subscribed_cities(path: Path) -> list[str]:
    initialise(path)
    with sqlite3.connect(path) as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT city FROM alert_subscriptions WHERE confirmed=1 AND active=1"
            )
        ]


def templates() -> dict:
    return json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "alert_templates.json").read_text(
            encoding="utf-8"
        )
    )


def render_alert(
    city: str,
    forecast_date: str,
    level: str,
    probability: float,
    language: str,
    unsubscribe_url: str,
) -> dict:
    language = language if language in templates() else "en"
    data = templates()[language]
    summary = data["summary"].format(
        city=city, date=forecast_date, level=level, probability=probability
    )
    lines = [
        summary,
        "",
        *[f"• {tip}" for tip in data["tips"].get(level, [])],
        "",
        data["disclaimer"],
        "",
        f"Unsubscribe: {unsubscribe_url}",
    ]
    return {
        "subject": f"ClimateGuard: {level} heat risk for {city}",
        "body": "\n".join(lines),
        "tips": data["tips"].get(level, []),
    }


def send_email(recipient: str, message: dict) -> None:
    """Send by Resend or SMTP; DRY_RUN defaults to true and never sends."""
    if os.getenv("ALERT_DRY_RUN", "true").lower() == "true":
        logger.info(
            json.dumps(
                {
                    "event": "alert_dry_run",
                    "recipient_domain": recipient.rsplit("@", 1)[-1],
                    "subject": message["subject"],
                }
            )
        )
        return
    if os.getenv("RESEND_API_KEY"):
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}"},
            json={
                "from": os.getenv("ALERT_FROM_EMAIL", "ClimateGuard <alerts@example.com>"),
                "to": [recipient],
                "subject": message["subject"],
                "text": message["body"],
            },
            timeout=15,
        )
        response.raise_for_status()
        return
    email = EmailMessage()
    email["From"] = os.getenv("ALERT_FROM_EMAIL", "alerts@example.com")
    email["To"] = recipient
    email["Subject"] = message["subject"]
    email.set_content(message["body"])
    with smtplib.SMTP(
        os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587")), timeout=15
    ) as smtp:
        smtp.starttls()
        if os.getenv("SMTP_USERNAME"):
            smtp.login(os.environ["SMTP_USERNAME"], os.environ.get("SMTP_PASSWORD", ""))
        smtp.send_message(email)


def can_send(subscription: dict, level: str, now: datetime) -> bool:
    return RISK_RANK[level] >= RISK_RANK[subscription["minimum_risk"]] and not (
        22 <= now.astimezone(IST).hour or now.astimezone(IST).hour < 6
    )


def deliver_if_due(
    path: Path,
    subscription: dict,
    forecast_date: str,
    level: str,
    probability: float,
    now: datetime,
    sender=send_email,
) -> bool:
    if not can_send(subscription, level, now):
        return False
    initialise(path)
    with sqlite3.connect(path) as connection:
        duplicate = connection.execute(
            "SELECT 1 FROM alert_deliveries WHERE subscription_id=? AND forecast_date=? AND risk_level=? AND status='sent'",
            (subscription["id"], forecast_date, level),
        ).fetchone()
        daily_count = connection.execute(
            "SELECT COUNT(*) FROM alert_deliveries WHERE subscription_id=? AND substr(created_at, 1, 10)=? AND status='sent'",
            (subscription["id"], now.astimezone(IST).date().isoformat()),
        ).fetchone()[0]
        failed = connection.execute(
            "SELECT attempts, created_at FROM alert_deliveries WHERE subscription_id=? AND forecast_date=? AND risk_level=? AND status='failed'",
            (subscription["id"], forecast_date, level),
        ).fetchone()
    if duplicate or daily_count >= int(os.getenv("ALERT_DAILY_CAP", "2")):
        return False
    if failed:
        attempts, last_attempt = failed
        retry_after = datetime.fromisoformat(last_attempt) + timedelta(minutes=min(60, 2**attempts))
        if now < retry_after:
            return False
    message = render_alert(
        subscription["city"].title(),
        forecast_date,
        level,
        probability,
        subscription["language"],
        f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8001')}/api/unsubscribe/{subscription['token']}",
    )
    try:
        sender(subscription["email"], message)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO alert_deliveries(subscription_id, forecast_date, risk_level, status, created_at) VALUES (?, ?, ?, 'sent', ?)",
                (subscription["id"], forecast_date, level, now.isoformat()),
            )
        return True
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "event": "alert_delivery_failed",
                    "subscription_id": subscription["id"],
                    "error": str(exc),
                }
            )
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                """INSERT INTO alert_deliveries(subscription_id, forecast_date, risk_level, status, error, created_at)
                   VALUES (?, ?, ?, 'failed', ?, ?)
                   ON CONFLICT(subscription_id, forecast_date, risk_level, status)
                   DO UPDATE SET attempts=alert_deliveries.attempts + 1, error=excluded.error, created_at=excluded.created_at""",
                (subscription["id"], forecast_date, level, str(exc)[:300], now.isoformat()),
            )
        return False
