
"""
SAT Answer Grader — Fast Version
---------------------------------
The browser compresses each image and sends images in parallel,
one per request, to POST /api/grade_one.

Each request makes one Claude API call.

Extended Thinking:
- OFF: Standard Claude response.
- ON: Adaptive thinking with configurable reasoning effort.

The browser must send thinking=1 to enable adaptive thinking.
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
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener


# --------------------------------------------------
# IMAGE CONFIGURATION
# --------------------------------------------------

register_heif_opener()

MAX_IMAGES = 27
MAX_EDGE_PX = 1280
JPEG_QUALITY = 80


# --------------------------------------------------
# CLAUDE API CONFIGURATION
# --------------------------------------------------

MODEL = os.environ.get(
    "CLAUDE_MODEL",
    "claude-sonnet-5-5"
)

ANSWER_MAX_TOKENS = 600

# Maximum output allowance for requests using adaptive thinking.
# This is NOT a fixed thinking budget.
THINKING_MAX_TOKENS = 4600

# Reasoning effort: low, medium, or high.
THINKING_EFFORT = os.environ.get(
    "THINKING_EFFORT",
    "high"
).lower()

THINKING_TIMEOUT = 120

client = anthropic.Anthropic(
    max_retries=4,
    timeout=60
)


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    secrets.token_hex(32)
)

app.permanent_session_lifetime = 60 * 60 * 24 * 14

APP_PASSWORD = os.environ.get("APP_PASSWORD")


# --------------------------------------------------
# LOGIN PROTECTION
# --------------------------------------------------

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


# --------------------------------------------------
# CLAUDE SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """You are answering ONE photographed SAT-style multiple-choice question.

Respond with ONLY a JSON object (no prose, no markdown fences), keys in this order:

{"question_number": "3", "work": "...", "answer": "A", "answer_text": "..."}

Rules:
- "question_number": the question number printed on the page, digits only (e.g. "8"), or "?" if not visible. Read it exactly as printed; never guess or infer it from the order of the photos.
- "work": very brief scratch work, at most 60 words. For math, the key steps. For reading/writing, the deciding reason.
- "answer": a single capital letter (A, B, C, D, or E if present) for the correct choice.
- "answer_text": that option's text, shortened to under 12 words.
- If the question is illegible or ambiguous, use "answer": "?" and explain briefly in "answer_text".
"""


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def to_small_jpeg(raw: bytes) -> bytes:
    """
    Compress an image into a smaller JPEG.

    If the browser already sends a reasonably small JPEG,
    skip image decoding to improve performance.
    """

    if not raw:
        raise ValueError("The uploaded image is empty.")

    if raw[:3] == b"\xff\xd8\xff" and len(raw) < 1_500_000:
        return raw

    img = Image.open(io.BytesIO(raw)).convert("RGB")

    longest = max(img.size)

    if longest > MAX_EDGE_PX:

        scale = MAX_EDGE_PX / longest

        new_size = (
            int(img.width * scale),
            int(img.height * scale)
        )

        img = img.resize(
            new_size,
            Image.LANCZOS
        )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY
    )

    return buffer.getvalue()


# --------------------------------------------------
# JSON RESPONSE PARSER
# --------------------------------------------------

def parse_model_json(text: str):
    """
    Extract the JSON object from Claude's response.
    """

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError(
            "No valid JSON object found.",
            text,
            0
        )

    return json.loads(text[start:end + 1])


# --------------------------------------------------
# LOGIN ROUTE
# --------------------------------------------------

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
        error=error
    )


# --------------------------------------------------
# LOGOUT ROUTE
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

@app.route("/")
@login_required
def index():

    return render_template(
        "index.html",
        max_images=MAX_IMAGES
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    return "ok"


# --------------------------------------------------
# SAT QUESTION GRADING API
# --------------------------------------------------

@app.route("/api/grade_one", methods=["POST"])
@login_required
def grade_one():

    # Retrieve uploaded image.
    f = request.files.get("image")

    if not f:
        return jsonify({
            "error": "No image received."
        }), 400

    # Check whether Extended Thinking is enabled.
    use_thinking = request.form.get("thinking") == "1"

    raw_text = ""

    try:

        # ------------------------------------------
        # IMAGE PROCESSING
        # ------------------------------------------

        raw_image = f.read()

        data = raw_image

        encoded_image = base64.b64encode(
            data
        ).decode("utf-8")


        # ------------------------------------------
        # CLAUDE API REQUEST
        # ------------------------------------------

        kwargs = {

            "model": MODEL,

            "max_tokens": ANSWER_MAX_TOKENS,

            "system": SYSTEM_PROMPT,

            "messages": [
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "image",

                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_image
                            }
                        },

                        {
                            "type": "text",
                            "text": "Answer this question. JSON only."
                        }

                    ]
                }
            ]
        }


        # ------------------------------------------
        # ADAPTIVE THINKING CONFIGURATION
        # ------------------------------------------

        api = client

        if use_thinking:

            kwargs["max_tokens"] = THINKING_MAX_TOKENS

            kwargs["thinking"] = {
                "type": "adaptive"
            }

            kwargs["output_config"] = {
                "effort": THINKING_EFFORT
            }

            api = client.with_options(
                timeout=THINKING_TIMEOUT
            )


        # ------------------------------------------
        # SEND REQUEST TO CLAUDE
        # ------------------------------------------

        response = api.messages.create(**kwargs)


        # ------------------------------------------
        # EXTRACT TEXT RESPONSE
        # ------------------------------------------

        raw_text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )


        # ------------------------------------------
        # PARSE JSON
        # ------------------------------------------

        result = parse_model_json(raw_text)


        # ------------------------------------------
        # RETURN RESULT
        # ------------------------------------------

        return jsonify({
            "result": result
        })


    # ----------------------------------------------
    # ERROR HANDLING
    # ----------------------------------------------

    except json.JSONDecodeError:

        return jsonify({
            "error": "Model response wasn't valid JSON.",
            "raw": raw_text
        }), 502

    except UnidentifiedImageError:

        return jsonify({
            "error": "The uploaded file is not a supported image."
        }), 400

    except anthropic.APIError as e:

        return jsonify({
            "error": f"Claude API error: {e}"
        }), 502

    except Exception as e:

        app.logger.exception("Unexpected grading error")

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
