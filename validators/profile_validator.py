def validate_profile(profile: dict) -> tuple[bool, list[str]]:
    """
    Validates that the profile has all required fields for ML inference.
    Returns (is_valid, missing_fields)
    """
    required_fields = ['age', 'weight_kg', 'height_cm', 'gender', 'fitness_level']
    missing = [f for f in required_fields if not profile.get(f)]
    return len(missing) == 0, missing


def validate_numeric_ranges(profile: dict) -> tuple[bool, str]:
    """
    Validates that numeric fields are within realistic ranges.
    Returns (is_valid, error_message)
    """
    age = profile.get('age', 0)
    weight = profile.get('weight_kg', 0)
    height = profile.get('height_cm', 0)

    if not (10 <= age <= 100):
        return False, "Age must be between 10 and 100."
    if not (30 <= weight <= 250):
        return False, "Weight must be between 30kg and 250kg."
    if not (100 <= height <= 250):
        return False, "Height must be between 100cm and 250cm."

    return True, ""