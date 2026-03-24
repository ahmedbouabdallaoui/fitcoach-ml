from flask import Blueprint, request, jsonify
from security.service_token import require_service_token
from security.payload_decryptor import decrypt_payload
from validators.profile_validator import validate_profile, validate_numeric_ranges
from validators.goal_validator import validate_goal
from services.training_plan_service import build_plan
from services.groq_service import format_training_plan

training_plan_bp = Blueprint('training_plan', __name__)


@training_plan_bp.route('/predict/training-plan', methods=['POST'])
@require_service_token
@decrypt_payload
def predict_training_plan():
    data = request.decrypted_data

    profile = {
        'age': data.get('age'),
        'weight_kg': data.get('weight_kg'),
        'height_cm': data.get('height_cm'),
        'gender': data.get('gender'),
        'fitness_level': data.get('fitness_level'),
        'body_fat_percentage': data.get('body_fat_percentage')
    }

    context = {
        'goal': data.get('goal'),
        'days_per_week': data.get('days_per_week', 3),
        'duration_weeks': data.get('duration_weeks', 4),
        'specific_days': data.get('specific_days')
    }

    # --- Validate profile ---
    is_valid, missing = validate_profile(profile)
    if not is_valid:
        return jsonify({'error': f'Missing profile fields: {missing}'}), 400

    ranges_valid, range_error = validate_numeric_ranges(profile)
    if not ranges_valid:
        return jsonify({'error': range_error}), 400

    # --- Validate goal ---
    goal_validation = validate_goal(profile, context['goal'])

    # --- Build plan ---
    plan = build_plan(profile, context)

    # --- Format with Groq ---
    user_name = data.get('user_name', 'there')
    formatted = format_training_plan(user_name, plan)

    return jsonify({
        'plan_type': plan['plan_type'],
        'goal': plan['goal'],
        'fitness_level': plan['fitness_level'],
        'days_per_week': plan['days_per_week'],
        'duration_weeks': plan['duration_weeks'],
        'weekly_plan': plan['weekly_plan'],
        'formatted_plan': formatted,
        'goal_valid': goal_validation['goal_valid'],
        'recommended_goal': goal_validation['recommended_goal'],
        'warning_message': goal_validation['warning_message']
    })