import os
from functools import wraps
from flask import request, jsonify


# Protects all ML endpoints from unauthorized access.
# Every request from .NET must include the INTERNAL_SERVICE_KEY in the header.
# If the key is missing or wrong, the request is rejected with 401.
def require_service_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Service-Token")

        if not token:
            return jsonify({"error": "Missing service token"}), 401

        if token != os.environ.get("INTERNAL_SERVICE_KEY"):
            return jsonify({"error": "Invalid service token"}), 401

        return f(*args, **kwargs)

    return decorated