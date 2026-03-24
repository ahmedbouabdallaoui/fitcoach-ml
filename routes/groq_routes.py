from flask import Blueprint, request, jsonify
from security.service_token import require_service_token
from security.payload_decryptor import decrypt_payload
from services.groq_service import (
    generate_profile_question,
    generate_context_question,
    extract_profile_data,
    extract_context_data,
    format_goal_warning,
    generate_rag_response
)

groq_bp = Blueprint('groq', __name__)


@groq_bp.route('/groq/profile-question', methods=['POST'])
@require_service_token
@decrypt_payload
def profile_question():
    data = request.decrypted_data
    result = generate_profile_question(
        data.get('user_name', 'there'),
        data.get('missing_fields', [])
    )
    return jsonify({'response': result})


@groq_bp.route('/groq/context-question', methods=['POST'])
@require_service_token
@decrypt_payload
def context_question():
    data = request.decrypted_data
    result = generate_context_question(
        data.get('user_name', 'there'),
        data.get('missing_fields', []),
        data.get('tag', 'training')
    )
    return jsonify({'response': result})


@groq_bp.route('/groq/extract-profile', methods=['POST'])
@require_service_token
@decrypt_payload
def extract_profile():
    data = request.decrypted_data
    result = extract_profile_data(
        data.get('message', ''),
        data.get('missing_fields', [])
    )
    return jsonify(result)


@groq_bp.route('/groq/extract-context', methods=['POST'])
@require_service_token
@decrypt_payload
def extract_context():
    data = request.decrypted_data
    result = extract_context_data(
        data.get('message', ''),
        data.get('missing_fields', []),
        data.get('tag', 'training')
    )
    return jsonify(result)


@groq_bp.route('/groq/goal-warning', methods=['POST'])
@require_service_token
@decrypt_payload
def goal_warning():
    data = request.decrypted_data
    result = format_goal_warning(
        data.get('user_name', 'there'),
        data.get('warning'),
        data.get('recommended_goal', '')
    )
    return jsonify({'response': result})