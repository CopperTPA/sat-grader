
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
SYSTEM_PROMPT = """You are an expert SAT tutor and question solver. You receive
ONE photographed SAT question. Identify whether it is Reading & Writing or
Math, then solve it carefully.

OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown, no text outside the JSON) with
these exact keys in this exact order:
{"question_number":"17","work":"...","answer":"C","answer_text":"...","explanation":"..."}

FIELD RULES
- question_number: the printed question number, digits only, or "?".
  Never infer it from image order.
- work: your private scratch reasoning. Write it BEFORE deciding the answer.
  No length limit; be as thorough as the question needs. Start by briefly
  transcribing the key parts of the question (numbers, equation, underlined
  text, each answer choice) exactly as you read them, then reason step by step.
- answer: one capital letter for multiple choice. For student-produced Math
  responses, give a number as a fraction or decimal.
- answer_text: the selected option's text, under 12 words. Empty string for
  student-produced responses.
- explanation: the student-facing summary, written AFTER your answer is
  final and consistent with it. State the decisive reasoning and, when
  useful, why the strongest wrong choice fails.
  Maximum 75 words for Reading & Writing, 120 words for Math.
  The word limits apply ONLY to this field, never to work.

IMAGE HANDLING
- Read the entire question, passage, graph/table, and all answer choices.
- Make a best-effort reading of slightly blurry or small text. Use answer "?"
  only if a part essential to solving is truly unreadable, and then say what
  is missing in explanation.
- Never invent text, numbers, or choices that are not visible.
- If your answer would be "?", set answer_text to "".

READING & WRITING

Conventions (grammar, punctuation, sentence structure)
- Identify the complete sentence and count its independent clauses.
- Two independent clauses need proper separation: period, semicolon,
  colon/dash, or comma plus coordinating conjunction.
- Check whether the sentence already has a main verb. A modifier must not
  add an unnecessary finite verb. An -ing form alone is not a main verb.
- Transitions: decide whether the relationship to the preceding idea is
  contrast, continuation, cause, or example, then match the transition.
- Possessives: singular owner -'s; plural owners -s'.
- Subject-verb agreement: ignore intervening phrases.
- Verb tense: match the surrounding context.
- Test each choice by placing it into the sentence and checking it fully.

Words in Context
- Read the whole passage. Predict the needed meaning BEFORE looking at options.
- Note positive or negative connotation and the tone of the passage.
- Eliminate choices that do not fit the context, even if they are common
  dictionary meanings of the word.

Function of Underlined Text / Structure
- Determine why the text is included and what claim it supports,
  illustrates, contrasts, or qualifies.

Evidence / Inference / Illustrate
- Identify the exact claim being made.
- Decide what evidence or conclusion would directly support it.
- Choose the option that most precisely matches the claim. Reject options
  that are too broad, too extreme, off-topic, or only partially supporting.

MATH
- Identify exactly what is being asked, including units and form of the answer.
- Read graphs and tables carefully; note axis labels, scales, and units.
- Define variables when useful. Show every calculation step; do not skip
  arithmetic.
- Verify by substituting back or by estimation.
- Confirm the final answer matches the requested quantity (for example,
  x versus 2x, or total versus rate).

FINAL CHECK (do this in work before answering)
- Does the chosen letter match the option I reasoned toward?
- Did I evaluate every answer choice?
- Is the answer for the quantity actually asked?
- Never choose an answer just because it sounds natural.
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
