import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Groq client ---
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = 'llama-3.3-70b-versatile'


def _chat(prompt: str, system: str = None) -> str:
    """Base Groq call — returns raw text response."""
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content


def _chat_json(prompt: str) -> dict:
    """Groq call that returns parsed JSON — used for data extraction."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,  # low temperature for structured extraction
        max_tokens=500
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown if present
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ============================================================
# EXTRACTION FUNCTIONS — low temperature, structured output
# ============================================================

def extract_profile_data(message: str, missing_fields: list[str]) -> dict:
    """
    Extracts profile data from a user message.
    Only extracts fields that are in missing_fields.
    """
    prompt = f"""
Extract user profile information from this message.
Only extract these specific fields: {', '.join(missing_fields)}

Return ONLY a valid JSON object. Use null for fields not mentioned.
No markdown, no explanation, just JSON.

Field rules:
- age: integer (years)
- weight_kg: float (kilograms)
- height_cm: float (centimeters)
- gender: "Male" or "Female" only
- fitness_level: "beginner", "intermediate", or "advanced" only
- body_fat_percentage: float (percentage, optional)

User message: "{message}"

JSON:
"""
    try:
        result = _chat_json(prompt)
        # Only return non-null values
        return {k: v for k, v in result.items() if v is not None}
    except Exception:
        return {}


def extract_context_data(message: str, missing_fields: list[str], tag: str) -> dict:
    """
    Extracts conversation context data from a user message.
    """
    prompt = f"""
Extract training information from this message for a {tag} plan.
Only extract these specific fields: {', '.join(missing_fields)}

Return ONLY a valid JSON object. Use null for fields not mentioned.
No markdown, no explanation, just JSON.

Field rules:
- goal: "weight_loss", "muscle_gain", "endurance", or "maintenance" only
- days_per_week: integer (1-7)
- specific_days: list of day names ["Monday", "Tuesday"...] or null
- duration_weeks: integer (weeks)
- activity_level: "sedentary", "light", "moderate", "active", or "very_active" only
- weekly_training_hours: float (hours per week)
- previous_injuries: true or false
- recent_symptoms: string describing symptoms or null

User message: "{message}"

JSON:
"""
    try:
        result = _chat_json(prompt)
        return {k: v for k, v in result.items() if v is not None}
    except Exception:
        return {}


# ============================================================
# QUESTION GENERATION — natural, friendly tone
# ============================================================

def generate_profile_question(user_name: str, missing_fields: list[str]) -> str:
    """Generates a natural question to collect missing profile fields."""
    fields_str = ', '.join(missing_fields)
    prompt = f"""
You are FitCoach, a friendly and motivating AI fitness trainer.
Generate a short, natural and friendly message asking {user_name} for their: {fields_str}

Rules:
- Keep it conversational and warm
- Ask for all missing fields in one message
- Don't use bullet points or lists
- Maximum 2 sentences
- Don't repeat the user's name more than once
"""
    return _chat(prompt)


def generate_context_question(user_name: str, missing_fields: list[str], tag: str) -> str:
    """Generates a natural question to collect missing context fields."""
    fields_str = ', '.join(missing_fields)
    prompt = f"""
You are FitCoach, a friendly AI fitness trainer helping {user_name} with their {tag} plan.
Generate a short, natural question asking for: {fields_str}

Rules:
- Keep it conversational and warm
- Ask for all missing fields in one message
- Don't use bullet points
- Maximum 2 sentences
"""
    return _chat(prompt)


# ============================================================
# FORMATTING FUNCTIONS — high temperature, natural language
# ============================================================

def format_goal_warning(user_name: str, warning: str, recommended_goal: str) -> str:
    """Formats a goal reality check warning empathetically."""
    prompt = f"""
You are FitCoach, a caring and honest AI fitness trainer.
{user_name} has set a goal that may not be optimal for their profile.

Warning to communicate: {warning}
Recommended goal: {recommended_goal}

Write a short, empathetic message that:
- Gently explains why their goal might not be ideal
- Recommends the better goal
- Asks if they want to proceed with their original goal anyway
- Is warm and encouraging, not judgmental
- Maximum 3 sentences
"""
    return _chat(prompt)


def format_training_plan(user_name: str, plan: dict) -> str:
    """Formats a training plan into natural motivating language."""
    plan_summary = json.dumps(plan, indent=2)
    prompt = f"""
You are FitCoach, an expert and motivating AI fitness trainer.
Format this training plan for {user_name} in a clear, motivating and structured way.

Plan data:
{plan_summary}

Requirements:
- Start with one encouraging intro sentence
- For each workout day use this EXACT format:

**[Day] — [Muscle Focus] ([Duration] mins)**
1. [Exercise Name] ([classification]) — [sets]x[reps] | Rest: [rest]s
   💡 Tip: [one short coaching tip for compound exercises only]
2. [Exercise Name] ([classification]) — [sets]x[reps] | Rest: [rest]s
...

- After all workout days add one line: 🛌 Rest days: [list rest days]
- End with one short motivating sentence
- No nested bullet points
- No extra indentation
- Maximum 600 words
"""
    return _chat(prompt)


def format_nutrition_advice(user_name: str, nutrition: dict) -> str:
    """Formats nutrition advice into clear, educational language."""
    nutrition_summary = json.dumps(nutrition, indent=2)
    prompt = f"""
You are FitCoach, an expert AI nutrition coach.
Format this nutrition plan for {user_name} in a clear and educational way.

Nutrition data:
{nutrition_summary}

Requirements:
- Start with their daily calorie target and goal
- Explain BMR and TDEE briefly in simple terms
- Present macro breakdown clearly (protein/carbs/fats in grams)
- For each meal include:
  * Timing and calories
  * Focus/goal of the meal
  * 3-4 specific food suggestions commonly used by gym people
    (e.g. chicken breast, brown rice, oats, eggs, whey protein, sweet potato, 
    Greek yogurt, almonds, banana, tuna, cottage cheese, whole grain bread)
- Mention supplement suggestions with brief reasons
- End with 3 practical eating tips
- Use emojis sparingly
- Maximum 600 words
"""
    return _chat(prompt)


def format_injury_report(user_name: str, injury_data: dict) -> str:
    """Formats injury prediction into empathetic, actionable language."""
    injury_summary = json.dumps(injury_data, indent=2)
    prompt = f"""
You are FitCoach, a caring AI fitness trainer focused on injury prevention.
Format this injury risk report for {user_name} in an empathetic and actionable way.

Injury data:
{injury_summary}

Requirements:
- Start by acknowledging their concern
- Present the risk score clearly with context (what it means)
- List risk factors in plain language
- Provide 3-5 specific, actionable prevention recommendations
- If high risk (score >= 7), strongly recommend consulting a professional
- End with an encouraging message
- Be empathetic, not alarming
- Maximum 400 words
"""
    return _chat(prompt)


def generate_rag_response(message: str, history: list[dict], profile: dict) -> str:
    """
    Generates a RAG-style response for general fitness questions.
    Uses conversation history for context.
    """
    # Build conversation context
    history_str = '\n'.join([
        f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
        for msg in history[-6:]  # last 6 messages for context
        if msg.get('content')
    ])

    profile_str = f"Age: {profile.get('age')}, Weight: {profile.get('weight_kg')}kg, Level: {profile.get('fitness_level')}"

    system = f"""You are FitCoach, an expert AI fitness and nutrition assistant for FitUnity platform.
You help users with fitness questions, platform navigation, exercise advice and nutrition guidance.

User profile: {profile_str}

Always:
- Give specific, evidence-based advice
- Be encouraging and motivating
- Keep responses concise and actionable
- If asked about the platform, guide them to the right feature
"""

    prompt = f"""
Conversation history:
{history_str}

Current question: {message}

Answer helpfully and concisely.
"""
    return _chat(prompt, system=system)