"""
SAT Answer Grader — Flask backend
----------------------------------
Accepts up to 27 photographed SAT-style question images in a single request,
sends them to the Claude API in one call, and returns the correct answer
letter (plus a short justification) for each question, in order.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."
    python app.py

Then open http://localhost:5000 on your phone (same wifi) or deploy this
behind a real host (Render, Fly.io, a VPS, etc.) for a public mobile site.

The frontend lives in templates/index.html and posts to POST /api/grade.
"""

import base64
import functools
import io
import json
import os
import re
import secrets

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image
from pillow_heif import register_heif_opener
import anthropic

register_heif_opener()  # lets Pillow open .heic/.heif files straight from an iPhone

MAX_IMAGES = 27
MAX_EDGE_PX = 1568        # Claude downsamples past this anyway; resizing first saves tokens + time
JPEG_QUALITY = 85
MODEL = "claude-sonnet-5"  # use "claude-opus-5-5" instead if you want more accuracy on hard questions

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = 60 * 60 * 24 * 14  # stay logged in for 2 weeks

APP_PASSWORD = os.environ.get("APP_PASSWORD")  # set this in your host's env vars, never in code

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

SYSTEM_PROMPT = """You are grading a set of photographed SAT-style multiple-choice \
questions, one question per image, given in a fixed order.

For every image, work out the correct answer to the question shown.

Respond with ONLY a JSON array (no prose, no markdown fences) with one object \
per image, in the same order as the images:

[
  {"image": 1, "question_number": "3", "answer": "A", "answer_text": "short paraphrase of option A"},
  ...
]

Rules:
- "image" is the 1-based position of the image in this request.
- "question_number" is the number printed on the page, as a string. Use "?" if not visible.
- "answer" is a single capital letter matching the correct choice (A, B, C, D, or E if present).
- "answer_text" is the text of that chosen option, shortened to under 12 words.
- If a question is illegible or ambiguous, still return an object for it with "answer": "?" \
and a brief note in "answer_text".
- Return exactly one object per image, in order. Do not skip any, and do not merge two images \
into one object.
"""


def resize_for_upload(file_storage) -> bytes:
    """Downscale + re-encode an uploaded photo to keep tokens and latency low."""
    img = Image.open(file_storage.stream).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def build_content_blocks(files):
    blocks = []
    for i, f in enumerate(files, start=1):
        blocks.append({"type": "text", "text": f"Image {i}:"})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(resize_for_upload(f)).decode("utf-8"),
            },
        })
    blocks.append({
        "type": "text",
        "text": (
            f"There are {len(files)} images above, numbered 1 to {len(files)}. "
            "Return the JSON array described in your instructions."
        ),
    })
    return blocks


def parse_model_json(text: str):
    """Claude is told to return raw JSON, but strip fences defensively just in case."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return "APP_PASSWORD isn't set on the server yet — add it in your host's environment variables.", 500

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


@app.route("/api/grade", methods=["POST"])
@login_required
def grade():
    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No images received."}), 400
    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Send at most {MAX_IMAGES} images per request."}), 400

    raw_text = ""
    try:
        content = build_content_blocks(files)
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text")
        results = parse_model_json(raw_text)
    except json.JSONDecodeError:
        return jsonify({"error": "Model response wasn't valid JSON.", "raw": raw_text}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"Claude API error: {e}"}), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
