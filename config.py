SENDER_EMAIL = "youraccount@gmail.com"
APP_PASSWORD = "xxxx xxxx xxxx xxxx"     # 16-char Gmail app password, not your real password
RECEIVER_EMAIL = "security_guard@gmail.com"        # one address, or several separated by commas: "admin@x.com, guard@x.com"

# ----- SMS alerts (optional; needs an SMS provider account, Twilio is supported) -----
SMS_ENABLED = False
TWILIO_SID = ""                 # Account SID from the Twilio console
TWILIO_TOKEN = ""               # Auth Token from the Twilio console (keep this private)
TWILIO_FROM = ""                # your Twilio number like "+15551234567", or a Messaging Service SID (starts with MG)
SMS_RECIPIENTS = []             # full international format, e.g. ["+919876543210", "+919812345678"]
SMS_MIN_LEVEL = "Critical"      # "High" or "Critical": SMS costs money, so only serious incidents by default
SMS_COOLDOWN = 120              # seconds between SMS for the same camera

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "guard": {"password": "guard123", "role": "guard"},
}

# Live monitor camera list — edit this for real deployment.
# source: 0/1/2... for local USB webcams, or a full RTSP/HTTP URL string for phone/CCTV.
# location: "atm"/"gate"/"server_room" gets a higher risk multiplier; anything else is "general".
# time_of_day: "day" or "night" — or set to "auto" to use the system clock automatically.
CAMERAS = [
    {"id": 1, "source": 0, "location": "general", "time_of_day": "auto"},
    # {"id": 2, "source": "rtsp://user:pass@192.168.1.10:554/stream", "location": "atm", "time_of_day": "auto"},
    # {"id": 3, "source": "http://192.168.1.5:8080/video", "location": "gate", "time_of_day": "auto"},
]

# ----- Telegram alerts (free, no registration; the easiest way to alert a phone) -----
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = " "         # from @BotFather in Telegram: send /newbot and copy the token it gives you
TELEGRAM_CHAT_IDS = []          # a person's id or a group's id (negative number); use Settings -> "Find chat IDs", e.g. [123456789, -100123456789]
TELEGRAM_MIN_LEVEL = "High"     # "High" or "Critical"
TELEGRAM_COOLDOWN = 60          # seconds between messages for the same camera
TELEGRAM_SEND_CLIP = False     # True = also send the FACE-BLURRED evidence clip (it is then stored on Telegram's servers)
