import os
import base64
import json
from functools import wraps
from flask import request, jsonify
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# Loads the encryption key once at startup from environment variables.
def _load_key() -> bytes:
    raw_key = os.environ.get("ENCRYPTION_KEY")
    if not raw_key:
        raise ValueError("ENCRYPTION_KEY is missing from environment variables.")
    return base64.b64decode(raw_key)


# Cached key — loaded once, reused for every request.
_KEY = _load_key()


def _decrypt(encrypted_base64: str) -> dict:
    raw = base64.b64decode(encrypted_base64)

    # Split combined bytes back into their parts
    nonce       = raw[:12]
    tag         = raw[12:28]
    ciphertext  = raw[28:]

    aesgcm = AESGCM(_KEY)
    plaintext = aesgcm.decrypt(nonce, ciphertext + tag, associated_data=None)

    return json.loads(plaintext.decode("utf-8"))


# Decorator — decrypts the request body before the route function runs.
# Usage:
#   @require_service_token
#   @decrypt_payload
#   def my_route():
#       data = request.decrypted_data  ← clean decrypted dict
def decrypt_payload(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        encrypted = request.json.get("data") if request.json else None

        if not encrypted:
            return jsonify({"error": "Missing encrypted payload"}), 400

        try:
            request.decrypted_data = _decrypt(encrypted)
        except Exception:
            return jsonify({"error": "Failed to decrypt payload"}), 400

        return f(*args, **kwargs)

    return decorated