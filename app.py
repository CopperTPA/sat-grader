
"""
SAT Answer Grader
-----------------
Receives one question image per request and sends it to Claude.
The frontend can submit multiple requests concurrently.
"""

import base64
import io
import json
import os
import secrets
import functools

import anthropic
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from PIL import Image
from pillow_heif import register_heif_opener


# Register HEIC/HEIF image support
register_heif_opener()


# Configuration
MAX_IMAGES = 27
MAX_EDGE_PX = 1280
JPEG_QUALITY = 80
MAX_UPLOAD_MB = 10

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    secrets.token_hex(32),
)
app.permanent_session_lifetime = 60 * 60 * 24 * 14
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

APP_PASSWORD = os.environ.get("APP_PASSWORD")

client = anthropic.Anthropic(
    max_retries=2,
    timeout=60,
)


# Authentication
def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({
                    "error": "Session expired. Reload and sign in again."
                }), 401

            return redirect(
                url_for("login", next=request.path)
            )

        return view(*args, **kwargs)

    return wrapped


# SAT grading instructions
SYSTEM_PROMPT = """You are an SAT question solver. You receive ONE
photographed SAT question. Identify whether it is Reading & Writing
or Math, then solve it.

Return ONLY a valid JSON object with these exact keys and order:
{"question_number":"17","work":"...","answer":"C","answer_text":"..."}

GENERAL RULES
- Read the entire question, passage, and all visible answer choices.
- Never invent text or information that cannot be read.
- Evaluate all answer choices before selecting an answer.
- Explain the decisive reasoning and, when useful, why the strongest
  incorrect alternative fails.
- Keep explanations concise and specific.
- question_number: printed question number, digits only, or "?".
  Never infer it from image order.
- answer: one capital letter for multiple choice. For student-produced
  Math responses, provide a fraction or decimal.
- answer_text: selected option text, under 12 words. Empty for
  student-produced responses.
- If the question is illegible or ambiguous, use answer "?" and
  briefly explain the issue in answer_text.

READING & WRITING

CONVENTIONS
- Identify the complete sentence and count its independent clauses.
- Two independent clauses require appropriate separation:
  period, semicolon, colon/dash, or comma plus coordinating conjunction.
- Determine whether the sentence already has a main verb.
  A modifier must not introduce an unnecessary finite verb.
- An -ing form alone is not a main verb.
- In-sentence transitions connect to the preceding sentence.
  Determine whether the relationship is contrast or continuation.
- Check possessives carefully: singular owner -'s; plural owners -s'.
- Match subject and verb, ignoring intervening phrases.
- Match verb tense to surrounding context.

WORDS IN CONTEXT
- Read the entire passage.
- Predict the meaning required by context before examining options.
- Determine positive or negative connotation when relevant.
- Eliminate choices that do not match the intended meaning.

FUNCTION OF UNDERLINED TEXT
- Determine why the text is included.
- Identify the claim it supports, illustrates, contrasts, or qualifies.

EVIDENCE / ILLUSTRATE
- Identify the claim.
- Determine what evidence would directly support it.
- Select the choice that demonstrates the claim.

MATH
- Identify exactly what is being asked, including units.
- Read graphs and tables carefully.
- Define variables when useful and solve accurately.
- Verify the result using substitution or estimation when appropriate.
- Ensure the final answer matches the requested quantity.

EXPLANATION LENGTH
- Reading & Writing: maximum 75 words.
- Math: maximum 120 words.

Never select an answer merely because it sounds natural.
"""

# Image processing
def to_small_jpeg(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")

    longest = max(img.size)

    if longest > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / longest

        new_size = (
            int(img.width * scale),
            int(img.height * scale),
        )

        img = img.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
    )

    return buffer.getvalue()


# JSON parsing
def parse_model_json(text: str):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise json.JSONDecodeError(
            "No JSON object found",
            text,
            0,
        )

    return json.loads(text[start:end + 1])


# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return "APP_PASSWORD isn't set on the server yet.", 500

    error = None

    if request.method == "POST":
        entered = request.form.get("password", "")

        if secrets.compare_digest(entered, APP_PASSWORD):
            session.clear()
            session["logged_in"] = True
            session.permanent = True

            return redirect(
                request.args.get("next") or url_for("index")
            )

        error = "Wrong password."

    return render_template(
        "login.html",
        error=error,
    )


# Logout route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Main page
@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        max_images=MAX_IMAGES,
    )


# Health check
@app.route("/health")
def health():
    return "ok"


# Grade one question
@app.route("/api/grade_one", methods=["POST"])
@login_required
def grade_one():
    uploaded_file = request.files.get("image")

    if not uploaded_file:
        return jsonify({
            "error": "No image received."
        }), 400

    raw_text = ""

    try:
        raw = uploaded_file.read()

        if not raw:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        image_data = to_small_jpeg(raw)

        image_base64 = base64.b64encode(
            image_data
        ).decode("utf-8")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1200,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {
                        "type": "ephemeral"
                    },
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Answer this question. JSON only.",
                        },
                    ],
                }
            ],
        )

        raw_text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        result = parse_model_json(raw_text)

        if not isinstance(result, dict):
            raise ValueError(
                "Model response must be a JSON object."
            )

        required_fields = {
            "question_number",
            "work",
            "answer",
            "answer_text",
        }

        if not required_fields.issubset(result):
            raise ValueError(
                "Model response is missing required fields."
            )

        if not all(
            isinstance(result[field], str)
            for field in required_fields
        ):
            raise ValueError(
                "Model response contains invalid field types."
            )

    except json.JSONDecodeError:
        return jsonify({
            "error": "Model response wasn't valid JSON.",
            "raw": raw_text,
        }), 502

    except anthropic.APIError as error:
        return jsonify({
            "error": f"Claude API error: {error}"
        }), 502

    except Exception as error:
        app.logger.exception("Question grading failed")

        return jsonify({
            "error": str(error)
        }), 500

    return jsonify({
        "result": result
    })


# Application entry point
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
