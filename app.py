import os, json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from google import genai
from google.genai import types

app = Flask(__name__)
handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])
line_config = Configuration(access_token=os.environ["LINE_ACCESS_TOKEN"])
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are a translation engine. Detect the language of the user's text.
If it is Thai, translate to English. If it is any other language, translate to Thai.

Return ONLY valid JSON in this exact shape:
{"formal": "...", "casual": "..."}

Rules for "formal":
- Professional register suitable for emails to a boss, clients, or business partners.
- Polite, clear, complete sentences. No slang, no contractions in English.
- In Thai, use polite particles (ครับ/ค่ะ) and formal vocabulary.

Rules for "casual":
- How a real person would text a friend. Natural, relaxed, idiomatic.
- Do NOT translate word-for-word. Rephrase so it sounds like something a native speaker would actually say.
- English: contractions, everyday phrasing. Thai: spoken style (เช่น "เดี๋ยว", "อ่ะ", "นะ", "แหละ") but not overdone.
- Keep the original meaning and tone (joking, annoyed, excited, etc.)."""

def translate(text: str) -> dict:
    resp = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    return json.loads(resp.text)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def on_message(event):
    try:
        result = translate(event.message.text)
        reply = f"📋 ทางการ\n{result['formal']}\n\n💬 แชทเพื่อน\n{result['casual']}"
    except Exception as e:
        print("ERROR:", e)
        reply = "ขอโทษครับ แปลไม่สำเร็จ ลองใหม่อีกครั้งนะครับ"

    with ApiClient(line_config) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)],
            )
        )

if __name__ == "__main__":
    app.run(port=5000)
