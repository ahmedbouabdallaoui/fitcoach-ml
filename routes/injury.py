from flask import Blueprint, request, jsonify
from security.service_token import require_service_token
from security.payload_decryptor import decrypt_payload
from services.injury_service import predict_injury_risk
from services.groq_service import format_injury_report

injury_bp = Blueprint('injury', __name__)


@injury_bp.route('/predict/injury-risk', methods=['POST'])
@require_service_token
@decrypt_payload
def predict_injury():
    data = request.decrypted_data

    profile = {
        'age': data.get('age'),
        'weight_kg': data.get('weight_kg'),
        'height_cm': data.get('height_cm'),
        'fitness_level': data.get('fitness_level')
    }

    context = {
        'weekly_training_hours': data.get('weekly_training_hours', 5),
        'has_previous_injuries': data.get('has_previous_injuries', False),
        'recent_symptoms': data.get('recent_symptoms')
    }

    result = predict_injury_risk(profile, context)

    user_name = data.get('user_name', 'there')
    formatted = format_injury_report(user_name, result)

    return jsonify({
        'risk_score': result['risk_score'],
        'risk_level': result['risk_level'],
        'high_risk': result['high_risk'],
        'risk_factors': result['risk_factors'],
        'formatted_report': formatted
    })