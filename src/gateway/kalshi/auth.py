"""Kalshi API authentication helpers.

RSA-PSS signing for REST and WebSocket authentication.
"""

from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def load_private_key(path: str):
    """Load an RSA private key from a PEM file."""
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def sign_pss(private_key, text: str) -> str:
    """Sign text with RSA-PSS padding."""
    sig = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def auth_headers(key_id: str, private_key, method: str, path: str) -> dict:
    """Generate Kalshi authentication headers.

    The path should include /trade-api/v2 prefix for REST,
    or be /trade-api/ws/v2 for WebSocket.
    """
    ts = str(int(time.time() * 1000))
    # Strip query params for signing
    clean_path = path.split("?")[0]
    msg = ts + method + clean_path
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sign_pss(private_key, msg),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
