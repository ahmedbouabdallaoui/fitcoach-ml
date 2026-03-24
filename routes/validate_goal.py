from flask import Blueprint, request, jsonify
from security.service_token import require_service_token
from security.payload_decryptor import decrypt_payload
from validators.goal_validator import validate_goal

validate_goal_bp = Blueprint('validate_goal', __name__)


@validate_goal_bp.route('/predict/validate-goal', methods=['POST'])
@require_service_token
@decrypt_payload
def predict_validate_goal():
    data = request.decrypted_data

    profile = {
        'age': data.get('age'),
        'weight_kg': data.get('weight_kg'),
        'height_cm': data.get('height_cm'),
        'gender': data.get('gender'),
        'fitness_level': data.get('fitness_level'),
        'body_fat_percentage': data.get('body_fat_percentage')
    }

    goal = data.get('goal', 'maintenance')

    result = validate_goal(profile, goal)

    return jsonify(result)