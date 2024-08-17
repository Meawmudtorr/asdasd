from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
import linebot.v3.messaging
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    ImageMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
from linebot.v3.messaging.models.message import Message
from linebot.v3.messaging.models.quick_reply import QuickReply
from linebot.v3.messaging.models.sender import Sender
import numpy as np
from PIL import Image
import joblib
import requests
from io import BytesIO

app = Flask(__name__)

# ใส่ LINE credentials ของคุณ
line_bot_api = Configuration('XYwfM8fustXQYD2xnnsIk9FlQjavaaf0QkqeDZtu4DvtXAMHUaE1h2fZUF7hwMo4jST7BSUWWFWa90rGnIkGUMvZQUxwGfUu4dCIBuyAA0kQCFbpWqA/xVa0pxHiPiWBVVUZVCmTFWoDEg7+R9JLjgdB04t89/1O/w1cDnyilFU=')
handler = linebot.v3.WebhookHandler('cb2f94676b5546bd74374cee759ac5fb')

# โหลดโมเดลของคุณ
model = joblib.load('cr6_model.pkl')

def extract_rgb(image):
    rgb_mean = np.array(image).mean(axis=(0, 1))
    return rgb_mean

def predict_concentration(image):
    rgb = extract_rgb(image)
    prediction = model.predict([rgb])[0]
    return prediction

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image = Image.open(BytesIO(message_content.content))

    concentration = predict_concentration(image)
    response_text = f'Predicted Cr6+ concentration: {concentration:.2f} mg/L'

    line_bot_api.reply_message(
        event.reply_token,
        TextMessage(text=response_text)
    )

if __name__ == "__main__":
    app.run(port=8000)
