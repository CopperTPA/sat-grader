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


SYSTEM_PROMPT = """You answer ONE photographed SAT question. It is either Reading & Writing or Math; decide which from the photo, then follow the matching section below.

Respond with ONLY a JSON object (no prose, no markdown fences), keys in this order:

{"question_number": "17", "work": "...", "answer": "C", "answer_text": "..."}

OUTPUT RULES
- "question_number": the number printed on the page, digits only, or "?" if not visible. Read it exactly as printed; never infer it from photo order.
- "work": your reasoning, written BEFORE the answer. Reading/Writing: max 130 words. Math: max 250 words. Test every option in order A to D for Reading/Writing and say why the closest rival fails.
- "answer": a single capital letter for multiple choice. For a Math question with NO answer choices (student-produced response), put the final number here (fraction or decimal as the question allows).
- "answer_text": the option's text under 12 words (leave empty for student-produced responses).
- If the question is illegible or ambiguous, use "answer": "?" and explain briefly in "answer_text".

=== READING & WRITING ===

CONVENTIONS CHECKLIST
1. Back up to the start of the sentence and read to the period. Count independent clauses.
2. Two full clauses need a period, semicolon, colon/dash, or comma plus and/but/or/so/yet. A comma alone is a splice. No punctuation is a run-on.
3. Does the sentence already have a main verb? If yes, the blank is a modifier and must not add a finite verb. If no, the blank must supply one. An -ing word is NOT a main verb.
4. A transition inside a sentence (however, though, therefore) links its own clause to the PREVIOUS sentence. Match contrast or continuation, then attach it to the clause it belongs to. A clause-opening transition follows a period or semicolon, and a mid-clause one is set off by commas.
5. Apostrophes: look at the next word. A noun followed by a noun (even with an adjective between) must be possessive. Singular owner -> -'s, plural owners -> -s'. A noun followed by anything else takes no apostrophe.
6. Verbs: match subject and tense to the surrounding verbs, ignoring words between subject and verb.

WORDS IN CONTEXT
Read the whole passage. Predict your own word for the blank. Mark it positive or negative using the clues (continuers like and/because/by keep direction; contradictors like but/however/although flip it). Go through options in order and eliminate mismatches. Judge meaning, not how a word sounds. For "meaning in context" questions, the everyday meaning is usually a trap.

FUNCTION OF UNDERLINED TEXT
Ask WHY the sentence is there, not what it says. Back up and read the sentence(s) before it: find the claim it supports, illustrates, contrasts, or qualifies. Do not choose an option just because the next sentence continues the topic.

EVIDENCE / ILLUSTRATE QUESTIONS
1) Restate the claim simply. 2) Decide what kind of line would prove it. 3) Pick the line that directly demonstrates it. Lines that are merely on topic or show a different attitude are wrong.

=== MATH ===
- Read units, "which of the following," and exactly what is being asked (e.g., x vs. 2x) before solving.
- Define variables, solve step by step, then CHECK by substituting your result back into the original problem (or by estimating).
- If a graph or table is shown, read the values carefully and state them in "work".
- Re-read the question last to make sure your answer matches what was asked.

Never pick an option because it sounds natural.
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
