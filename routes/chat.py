from flask import Blueprint, request, jsonify
from security.service_token import require_service_token
from security.payload_decryptor import decrypt_payload
from services.groq_service import generate_rag_response

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/predict/chat', methods=['POST'])
@require_service_token
@decrypt_payload
def chat():
    data = request.decrypted_data

    message = data.get('message', '')
    history = data.get('history', [])
    profile = data.get('profile', {})

    response = generate_rag_response(message, history, profile)

    return jsonify({
        'response': response
    })