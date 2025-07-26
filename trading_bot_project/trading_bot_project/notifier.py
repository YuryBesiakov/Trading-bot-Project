"""Notification services.

Supports sending alerts via Telegram, Slack or email.  The notifier
reads its configuration from the YAML config and exposes a unified
``notify`` method that dispatches messages to all enabled channels.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .utils import setup_logger


logger = setup_logger(__name__)


class Notifier:
    """Send notifications to various messaging platforms."""

    def __init__(self, config: Dict[str, any]):
        self.cfg = config
        # Telegram
        telegram_cfg = config.get("telegram", {})
        self.telegram_enabled = telegram_cfg.get("enabled", False)
        self.telegram_bot_token = telegram_cfg.get("bot_token")
        self.telegram_chat_id = telegram_cfg.get("chat_id")
        # Slack
        slack_cfg = config.get("slack", {})
        self.slack_enabled = slack_cfg.get("enabled", False)
        self.slack_token = slack_cfg.get("token")
        self.slack_channel = slack_cfg.get("channel")
        self.slack_client: WebClient | None = None
        if self.slack_enabled and self.slack_token:
            self.slack_client = WebClient(token=self.slack_token)
        # Email
        email_cfg = config.get("email", {})
        self.email_enabled = email_cfg.get("enabled", False)
        self.smtp_server = email_cfg.get("smtp_server")
        self.smtp_port = email_cfg.get("smtp_port", 587)
        self.email_username = email_cfg.get("username")
        self.email_password = email_cfg.get("password")
        self.email_recipient = email_cfg.get("recipient")

    def notify(self, subject: str, message: str) -> None:
        """Send a notification message to all configured channels."""
        if self.telegram_enabled:
            self._notify_telegram(message)
        if self.slack_enabled:
            self._notify_slack(message)
        if self.email_enabled:
            self._notify_email(subject, message)

    def _notify_telegram(self, message: str) -> None:
        if not (self.telegram_bot_token and self.telegram_chat_id):
            logger.error("Telegram notification is enabled but token or chat_id is missing")
            return
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        data = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code != 200:
                logger.error("Failed to send Telegram message: %s", resp.text)
        except Exception as e:
            logger.exception("Telegram notification error: %s", e)

    def _notify_slack(self, message: str) -> None:
        if not (self.slack_client and self.slack_channel):
            logger.error("Slack notification is enabled but client or channel is missing")
            return
        try:
            self.slack_client.chat_postMessage(channel=self.slack_channel, text=message)
        except SlackApiError as e:
            logger.error("Failed to send Slack message: %s", e.response["error"])

    def _notify_email(self, subject: str, message: str) -> None:
        if not (self.smtp_server and self.email_username and self.email_password and self.email_recipient):
            logger.error("Email notification is enabled but SMTP or credentials are missing")
            return
        msg = MIMEMultipart()
        msg["From"] = self.email_username
        msg["To"] = self.email_recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
        except Exception as e:
            logger.exception("Failed to send email: %s", e)
