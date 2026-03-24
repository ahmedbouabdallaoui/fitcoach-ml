from flask import Blueprint, request, jsonify
from security.service_token import require_service_token
from security.payload_decryptor import decrypt_payload
from services.nutrition_service import NutritionService, NutritionRequest
from services.groq_service import format_nutrition_advice
import dataclasses

nutrition_bp = Blueprint('nutrition', __name__)
nutrition_service = NutritionService()


@nutrition_bp.route('/predict/nutrition', methods=['POST'])
@require_service_token
@decrypt_payload
def predict_nutrition():
    data = request.decrypted_data

    nutrition_request = NutritionRequest(
        age=data.get('age', 25),
        weight_kg=data.get('weight_kg', 70),
        height_cm=data.get('height_cm', 170),
        gender=data.get('gender', 'Male'),
        activity_level=data.get('activity_level', 'moderate'),
        goal=data.get('goal', 'maintenance')
    )

    result = nutrition_service.calculate(nutrition_request)
    nutrition_dict = dataclasses.asdict(result)

    user_name = data.get('user_name', 'there')
    formatted = format_nutrition_advice(user_name, nutrition_dict)

    return jsonify({
        'bmr': result.bmr,
        'tdee': result.tdee,
        'target_calories': result.target_calories,
        'macros': {
            'protein_g': result.macros.protein_g,
            'carbs_g': result.macros.carbs_g,
            'fats_g': result.macros.fats_g
        },
        'goal': result.goal,
        'meal_structure': result.meal_structure,
        'supplement_suggestions': result.supplement_suggestions,
        'formatted_advice': formatted
    })