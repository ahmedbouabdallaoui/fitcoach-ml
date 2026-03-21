import pandas as pd
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'training', 'data')

# --- Load dataset once at startup ---
exercises_df = pd.read_csv(os.path.join(DATA_DIR, 'megaGymDataset.csv'))

# --- Goal to workout type mapping ---
GOAL_TO_TYPE = {
    'weight_loss': 'Cardio',
    'muscle_gain': 'Strength',
    'endurance':   'Plyometrics',
    'maintenance': 'Stretching'
}

# --- Fitness level mapping ---
LEVEL_MAP = {
    'beginner':     'Beginner',
    'intermediate': 'Intermediate',
    'advanced':     'Expert'
}

# --- Muscle group splits per number of days ---
MUSCLE_SPLITS = {
    2: [
        ['Chest', 'Triceps', 'Shoulders', 'Abdominals'],
        ['Lats', 'Middle Back', 'Biceps', 'Quadriceps', 'Hamstrings']
    ],
    3: [
        ['Chest', 'Triceps', 'Shoulders'],
        ['Lats', 'Middle Back', 'Biceps'],
        ['Quadriceps', 'Hamstrings', 'Glutes', 'Calves']
    ],
    4: [
        ['Chest', 'Triceps'],
        ['Lats', 'Middle Back', 'Biceps'],
        ['Quadriceps', 'Hamstrings', 'Glutes'],
        ['Shoulders', 'Abdominals', 'Calves']
    ],
    5: [
        ['Chest', 'Triceps'],
        ['Lats', 'Middle Back'],
        ['Quadriceps', 'Hamstrings', 'Glutes'],
        ['Shoulders', 'Traps'],
        ['Biceps', 'Abdominals', 'Calves']
    ],
    6: [
        ['Chest', 'Triceps'],
        ['Lats', 'Middle Back'],
        ['Quadriceps', 'Glutes'],
        ['Shoulders', 'Traps'],
        ['Biceps', 'Forearms'],
        ['Hamstrings', 'Abdominals', 'Calves']
    ]
}

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def _classify_exercise(equipment: str) -> str:
    """Classifies exercise as compound or isolation based on equipment."""
    compound_equipment = ['Barbell', 'Dumbbell', 'Cable', 'Kettlebells', 'E-Z Curl Bar']
    for e in compound_equipment:
        if e.lower() in str(equipment).lower():
            return 'compound'
    return 'isolation'


def _pick_exercises_for_muscle(filtered_df, muscle: str, workout_type: str) -> list[dict]:
    """
    Picks 2 exercises per muscle — 1 compound first, 1 isolation second.
    Compound = heavier, fewer reps. Isolation = lighter, more reps.
    """
    muscle_df = filtered_df[filtered_df['BodyPart'] == muscle].copy()

    # Fallback without level filter
    if len(muscle_df) == 0:
        muscle_df = exercises_df[
            (exercises_df['Type'] == workout_type) &
            (exercises_df['BodyPart'] == muscle)
        ].copy()

    if len(muscle_df) == 0:
        return []

    muscle_df['Rating'] = muscle_df['Rating'].fillna(0)
    muscle_df['classification'] = muscle_df['Equipment'].apply(_classify_exercise)
    muscle_df = muscle_df.sort_values('Rating', ascending=False)

    compound = muscle_df[muscle_df['classification'] == 'compound']
    isolation = muscle_df[muscle_df['classification'] == 'isolation']

    selected = []

    # 1 compound — heavy
    if len(compound) > 0:
        ex = compound.iloc[0]
        selected.append({
            'name': ex['Title'],
            'body_part': ex['BodyPart'],
            'equipment': ex['Equipment'],
            'level': ex['Level'],
            'description': str(ex['Desc'])[:200],
            'classification': 'compound',
            'sets': 4,
            'reps': 8,
            'rest_seconds': 120,
            'estimated_minutes': 8,
            'rating': float(ex['Rating'] or 0)
        })

    # 1 isolation — lighter
    if len(isolation) > 0:
        ex = isolation.iloc[0]
        selected.append({
            'name': ex['Title'],
            'body_part': ex['BodyPart'],
            'equipment': ex['Equipment'],
            'level': ex['Level'],
            'description': str(ex['Desc'])[:200],
            'classification': 'isolation',
            'sets': 3,
            'reps': 12,
            'rest_seconds': 60,
            'estimated_minutes': 6,
            'rating': float(ex['Rating'] or 0)
        })

    # Fallback if only one type available
    if len(selected) < 2 and len(muscle_df) >= 2:
        for _, ex in muscle_df.head(2).iterrows():
            if ex['Title'] not in [s['name'] for s in selected]:
                selected.append({
                    'name': ex['Title'],
                    'body_part': ex['BodyPart'],
                    'equipment': ex['Equipment'],
                    'level': ex['Level'],
                    'description': str(ex['Desc'])[:200],
                    'classification': _classify_exercise(ex['Equipment']),
                    'sets': 3,
                    'reps': 10,
                    'rest_seconds': 90,
                    'estimated_minutes': 6,
                    'rating': float(ex['Rating'] or 0)
                })

    return selected


def select_exercises(
    goal: str,
    fitness_level: str,
    days_per_week: int,
    specific_days: list[str] = None
) -> list[dict]:
    """
    Selects exercises per muscle group split based on goal and fitness level.

    Two behaviors:
    - specific_days provided → user said "I'm free Monday and Wednesday"
                               → respect exact days, derive count from list
    - specific_days is None  → user said "I train 4 days"
                               → use first N days of week
    """
    workout_type = GOAL_TO_TYPE.get(goal, 'Strength')
    level = LEVEL_MAP.get(fitness_level, 'Beginner')

    # --- Filter by type and level ---
    filtered = exercises_df[
        (exercises_df['Type'] == workout_type) &
        (exercises_df['Level'] == level)
    ]

    # Fallback if not enough
    if len(filtered) < 10:
        filtered = exercises_df[exercises_df['Type'] == workout_type]

    # --- Determine workout days ---
    if specific_days:
        # User specified exact days — preserve week order
        workout_days = [d for d in DAYS_OF_WEEK if d in specific_days]
        rest_days = [d for d in DAYS_OF_WEEK if d not in specific_days]
        days_per_week = len(workout_days)
    else:
        # User said number of days — use first N days
        days_per_week = min(max(days_per_week, 2), 6)
        workout_days = DAYS_OF_WEEK[:days_per_week]
        rest_days = DAYS_OF_WEEK[days_per_week:]

    muscle_split = MUSCLE_SPLITS.get(days_per_week, MUSCLE_SPLITS[3])

    weekly_plan = []

    for i, day in enumerate(workout_days):
        target_muscles = muscle_split[i]
        day_label = ' + '.join(target_muscles)
        day_exercises = []

        for muscle in target_muscles:
            exercises = _pick_exercises_for_muscle(filtered, muscle, workout_type)
            day_exercises.extend(exercises)

        # Real duration: 5 min warmup + exercises + 5 min cooldown
        total_minutes = 10 + sum(ex['estimated_minutes'] for ex in day_exercises)

        weekly_plan.append({
            'day': day,
            'type': 'workout',
            'workout_type': workout_type,
            'muscle_focus': day_label,
            'estimated_duration_minutes': total_minutes,
            'exercises': day_exercises
        })

    for day in rest_days:
        weekly_plan.append({
            'day': day,
            'type': 'rest',
            'workout_type': 'Rest',
            'muscle_focus': 'Recovery',
            'estimated_duration_minutes': 0,
            'exercises': []
        })

    return weekly_plan


def get_body_parts(goal: str) -> list[str]:
    """Returns the primary body parts targeted for a given goal."""
    workout_type = GOAL_TO_TYPE.get(goal, 'Strength')
    filtered = exercises_df[exercises_df['Type'] == workout_type]
    return filtered['BodyPart'].unique().tolist()