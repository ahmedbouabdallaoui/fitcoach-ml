import os
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Register Routes ---
from routes.training_plan import training_plan_bp
from routes.validate_goal import validate_goal_bp
from routes.nutrition import nutrition_bp
from routes.injury import injury_bp
from routes.chat import chat_bp
from routes.groq_routes import groq_bp

app.register_blueprint(groq_bp)
app.register_blueprint(training_plan_bp)
app.register_blueprint(validate_goal_bp)
app.register_blueprint(nutrition_bp)
app.register_blueprint(injury_bp)
app.register_blueprint(chat_bp)

# Health check — used by Docker to verify the service is running
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "service": "fitcoach-ml"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8087))
    app.run(host="0.0.0.0", port=port, debug=False)