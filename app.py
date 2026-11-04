
"""
SAT Answer Grader
-----------------
Receives one question image per request and sends it to Claude.
The frontend can submit multiple requests concurrently.
"""

import base64
import functools
import io
import json
import os
import secrets
from urllib.parse import urlparse

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
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener


# ============================================================
# Image support
# ============================================================

register_heif_opener()


# ============================================================
# Configuration
# ============================================================

MAX_IMAGES = 27
MAX_EDGE_PX = 1280
JPEG_QUALITY = 80
MAX_UPLOAD_MB = 10

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

APP_PASSWORD = os.environ.get("APP_PASSWORD")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

app = Flask(__name__)

# In production, set FLASK_SECRET_KEY in your hosting environment.
# The fallback is intended for local development only.
if not SECRET_KEY:
    app.logger.warning(
        "FLASK_SECRET_KEY is not set. Sessions may be invalidated "
        "whenever the application restarts."
    )
    SECRET_KEY = secrets.token_hex(32)

app.secret_key = SECRET_KEY
app.permanent_session_lifetime = 60 * 60 * 24 * 14

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

client = anthropic.Anthropic(
    max_retries=2,
    timeout=60,
)


# ============================================================
# SAT grading instructions
# ============================================================

SYSTEM_PROMPT = """You answer ONE photographed SAT question. It is either Reading & Writing or Math; decide which from the photo, then follow the matching section below.

Respond with ONLY a JSON object (no prose, no markdown fences), keys in this order:

{"question_number": "17", "work": "...", "answer": "C", "answer_text": "..."}

OUTPUT RULES
- "question_number": The number printed on the page, digits only, or "?" if not visible. Never infer it from photo order.
- "work": Your reasoning, written BEFORE the answer. Reading/Writing: max 130 words. Math: max 250 words. Evaluate every visible answer choice in order and explain why the closest incorrect choice fails.
- "answer": A single capital letter for multiple-choice questions. For Math questions with NO answer choices (student-produced response), provide the final number as a fraction or decimal as appropriate.
- "answer_text": The selected option's text, under 12 words. Leave empty for student-produced responses.
- If the question is illegible or ambiguous, use "answer": "?" and explain briefly in "answer_text".
- Never invent text or information that cannot be read from the image.

=== READING & WRITING ===

CONVENTIONS CHECKLIST
1. Back up to the start of the sentence and read to the period. Count independent clauses.
2. Two full clauses need a period, semicolon, colon/dash, or comma plus and/but/or/so/yet. A comma alone is a splice. No punctuation is a run-on.
3. Does the sentence already have a main verb? If yes, the blank is a modifier and must not add a finite verb. If no, the blank must supply one. An -ing word is NOT a main verb.
4. A transition inside a sentence (however, though, therefore) links its own clause to the PREVIOUS sentence. Match contrast or continuation, then attach it to the clause it belongs to.
5. Apostrophes: look at the next word. A noun followed by a noun (even with an adjective between) must be possessive. Singular owner -> -'s, plural owners -> -s'.
6. Verbs: match subject and tense to the surrounding verbs, ignoring words between subject and verb.

WORDS IN CONTEXT
Read the whole passage. Predict your own word for the blank. Mark it positive or negative using the clues. Go through options in order and eliminate mismatches. Judge meaning, not how a word sounds.

FUNCTION OF UNDERLINED TEXT
Ask WHY the sentence is there, not what it says. Read the preceding sentence(s) to find the claim it supports, illustrates, contrasts, or qualifies.

EVIDENCE / ILLUSTRATE QUESTIONS
1. Restate the claim simply.
2. Decide what kind of line would prove it.
3. Pick the line that directly demonstrates it.

=== MATH ===
- Read units and exactly what is being asked (e.g., x vs. 2x) before solving.
- Define variables and solve step by step.
- Check your result by substitution or estimation.
- If a graph or table is shown, read its values carefully.
- Re-read the question to make sure your answer matches what was asked.

Never pick an option because it sounds natural.
"""


# ============================================================
# Authentication
# ============================================================

def is_safe_redirect(target):
    """Allow only local relative redirects."""
    if not target:
        return False

    parsed = urlparse(target)

    return (
        not parsed.scheme
        and not parsed.netloc
        and target.startswith("/")
        and not target.startswith("//")
    )


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


# ============================================================
# Image processing
# ============================================================

def to_small_jpeg(raw: bytes) -> bytes:
    """Decode an uploaded image, correct its orientation, and resize it."""
    with Image.open(io.BytesIO(raw)) as original:
        img = ImageOps.exif_transpose(original)
        img = img.convert("RGB")

        longest = max(img.size)

        if longest > MAX_EDGE_PX:
            scale = MAX_EDGE_PX / longest

            new_size = (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
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
            optimize=True,
        )

        return buffer.getvalue()


# ============================================================
# JSON parsing and validation
# ============================================================

def parse_model_json(text: str):
    """Extract and parse the JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError(
            "No JSON object found",
            text,
            0,
        )

    return json.loads(text[start:end + 1])


def validate_model_result(result):
    """Check that the model returned all required string fields."""
    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")

    required_fields = {
        "question_number",
        "work",
        "answer",
        "answer_text",
    }

    if not required_fields.issubset(result):
        raise ValueError("Model response is missing required fields.")

    for field in required_fields:
        if not isinstance(result[field], str):
            raise ValueError(
                f"Model response field '{field}' must be a string."
            )

    return result


# ============================================================
# Login route
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return "APP_PASSWORD is not configured on the server.", 503

    error = None

    if request.method == "POST":
        entered = request.form.get("password", "")

        if secrets.compare_digest(entered, APP_PASSWORD):
            session.clear()
            session["logged_in"] = True
            session.permanent = True

            next_url = request.args.get("next")

            if is_safe_redirect(next_url):
                return redirect(next_url)

            return redirect(url_for("index"))

        error = "Wrong password."

    return render_template("login.html", error=error)


# ============================================================
# Logout route
# ============================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ============================================================
# Main page
# ============================================================

@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        max_images=MAX_IMAGES,
    )


# ============================================================
# Health check
# ============================================================

@app.route("/health")
def health():
    return "ok", 200


# ============================================================
# Grade one question
# ============================================================

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

        # Convert the uploaded image to a resized JPEG.
        image_data = to_small_jpeg(raw)

        image_base64 = base64.b64encode(
            image_data
        ).decode("utf-8")

        # Send the image and use ephemeral caching for the
        # repeated system prompt.
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
                            "text": (
                                "Answer this question. "
                                "Return JSON only."
                            ),
                        },
                    ],
                }
            ],
        )

        # Extract text from the model response.
        raw_text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        if not raw_text.strip():
            raise ValueError("The model returned an empty response.")

        # Parse and validate the JSON.
        result = parse_model_json(raw_text)
        result = validate_model_result(result)

        # Record prompt-cache usage in server logs.
        usage = response.usage

        app.logger.info(
            "Claude token usage: input=%s, output=%s, "
            "cache_read=%s, cache_creation=%s",
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
        )

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({
            "error": "The uploaded file is not a valid supported image."
        }), 400

    except json.JSONDecodeError:
        app.logger.warning(
            "Claude returned invalid JSON: %s",
            raw_text[:1000],
        )

        return jsonify({
            "error": "The model response wasn't valid JSON."
        }), 502

    except anthropic.APIError as error:
        app.logger.exception("Claude API request failed")

        return jsonify({
            "error": "The grading service is temporarily unavailable."
        }), 502

    except ValueError as error:
        app.logger.warning("Invalid grading response: %s", error)

        return jsonify({
            "error": "The model returned an invalid grading response."
        }), 502

    except Exception:
        app.logger.exception("Question grading failed")

        return jsonify({
            "error": "An unexpected server error occurred."
        }), 500

    return jsonify({
        "result": result
    })


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
