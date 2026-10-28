"""
SAT Answer Grader — fast version
---------------------------------
The browser shrinks each photo, then sends all photos IN PARALLEL, one per
request, to POST /api/grade_one. Each request makes one small Claude call.
Total time ≈ the slowest single call (a few seconds) instead of the sum of
everything.
"""

import base64
import io
import json
import os
import re
import secrets
import functools

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image
from pillow_heif import register_heif_opener
import anthropic

register_heif_opener()

MAX_IMAGES = 27
MAX_EDGE_PX = 1280
JPEG_QUALITY = 80
# Verify this string against https://docs.claude.com (models overview).
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = 60 * 60 * 24 * 14

APP_PASSWORD = os.environ.get("APP_PASSWORD")

# max_retries covers 429/overloaded errors when 27 calls fire at once
client = anthropic.Anthropic(max_retries=4, timeout=60)


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Session expired. Reload and sign in again."}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


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


def to_small_jpeg(raw: bytes) -> bytes:
    """Skip decoding entirely when the browser already sent a small JPEG."""
    if raw[:3] == b"\xff\xd8\xff" and len(raw) < 1_500_000:
        return raw
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    longest = max(img.size)
    if longest > MAX_EDGE_PX:
        s = MAX_EDGE_PX / longest
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def parse_model_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no object", text, 0)
    return json.loads(text[start:end + 1])


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
            return redirect(request.args.get("next") or url_for("index"))
        error = "Wrong password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", max_images=MAX_IMAGES)


@app.route("/health")
def health():
    return "ok"  # point an uptime pinger here so the server never sleeps


@app.route("/api/grade_one", methods=["POST"])
@login_required
def grade_one():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image received."}), 400

    raw_text = ""
    try:
        data = to_small_jpeg(f.read())
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(data).decode("utf-8"),
                    }},
                    {"type": "text", "text": "Answer this question. JSON only."},
                ],
            }],
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text")
        result = parse_model_json(raw_text)
    except json.JSONDecodeError:
        return jsonify({"error": "Model response wasn't valid JSON.", "raw": raw_text}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"Claude API error: {e}"}), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
