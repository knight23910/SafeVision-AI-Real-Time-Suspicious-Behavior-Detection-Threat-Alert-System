import base64, json, os, smtplib, urllib.error, urllib.parse, urllib.request, uuid
import pyttsx3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER, SMTP_PORT = "smtp.gmail.com", 587


def _as_list(value):
    """Accepts 'a@x.com, b@y.com' or ['a@x.com', 'b@y.com'] and returns a clean list."""
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(i).strip() for i in items if str(i).strip()]  # str() so numeric Telegram chat ids work too


def send_email_alert(sender_email, app_password, receiver_email, subject, body):
    recipients = _as_list(receiver_email)
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = sender_email, ", ".join(recipients), subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg, to_addrs=recipients)


def send_sms_alert(account_sid, auth_token, from_number, recipients, body, base="https://api.twilio.com"):
    """Sends one SMS per recipient through Twilio's REST API. Returns (accepted_count, [error messages]).
    'Accepted' means Twilio took the message; it does not guarantee the phone received it."""
    url = f"{base}/2010-04-01/Accounts/{account_sid}/Messages.json"
    auth = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
    sender_field = "MessagingServiceSid" if from_number.startswith("MG") else "From"  # a Messaging Service SID or a phone number / sender ID
    accepted, errors = 0, []
    for to in _as_list(recipients):
        data = urllib.parse.urlencode({"To": to, sender_field: from_number, "Body": body}).encode()
        req = urllib.request.Request(url, data=data, headers={"Authorization": f"Basic {auth}"})
        try:
            urllib.request.urlopen(req, timeout=15).read()
            accepted += 1
        except urllib.error.HTTPError as e:
            try:
                reason = json.loads(e.read().decode()).get("message", e.reason)
            except Exception:
                reason = e.reason
            errors.append(f"{to}: {e.code} {reason}")
        except Exception as e:
            errors.append(f"{to}: {e}")
    return accepted, errors


TELEGRAM_API = "https://api.telegram.org"


def _post_form(url, fields, files=None, timeout=20):
    """POST form fields (and optionally files) and return the JSON answer. Standard library only."""
    if files:
        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
        for k, (fname, content, ctype) in files.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\nContent-Type: {ctype}\r\n\r\n'.encode() + content + b"\r\n")
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    else:
        data, headers = urllib.parse.urlencode(fields).encode(), {}
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout).read().decode())


def _telegram_call(token, method, fields, files=None, base=TELEGRAM_API):
    """Returns (answer, error_text); exactly one of them is None."""
    try:
        return _post_form(f"{base}/bot{token}/{method}", fields, files), None
    except urllib.error.HTTPError as e:
        try:
            return None, json.loads(e.read().decode()).get("description", str(e.reason))
        except Exception:
            return None, str(e.reason)
    except Exception as e:
        return None, str(e)


def send_telegram_message(token, chat_ids, text, base=TELEGRAM_API):
    """Sends a text to every chat id (a person, or a group). Returns (delivered_count, [error messages])."""
    delivered, errors = 0, []
    for cid in _as_list(chat_ids):
        res, err = _telegram_call(token, "sendMessage", {"chat_id": cid, "text": text}, base=base)
        if res and res.get("ok"):
            delivered += 1
        else:
            errors.append(f"{cid}: {err or res}")
    return delivered, errors


def send_telegram_video(token, chat_ids, path, caption="", base=TELEGRAM_API):
    """Sends a video file (Telegram's bot limit is 50 MB). Returns (delivered_count, [error messages])."""
    if os.path.getsize(path) > 49 * 1024 * 1024:
        return 0, ["clip is larger than Telegram's 50 MB bot limit"]
    with open(path, "rb") as f:
        content = f.read()
    delivered, errors = 0, []
    for cid in _as_list(chat_ids):
        res, err = _telegram_call(token, "sendVideo", {"chat_id": cid, "caption": caption, "supports_streaming": "true"},
                                  {"video": (os.path.basename(path), content, "video/mp4")}, base=base)
        if res and res.get("ok"):
            delivered += 1
        else:
            errors.append(f"{cid}: {err or res}")
    return delivered, errors


def telegram_find_chats(token, base=TELEGRAM_API):
    """Lists chats that recently wrote to the bot, so nobody has to dig chat ids out of raw JSON. Returns ({id: label}, error)."""
    res, err = _telegram_call(token, "getUpdates", {}, base=base)
    if err or not res or not res.get("ok"):
        return None, err or "no answer from Telegram"
    chats = {}
    for upd in res.get("result", []):
        for key in ("message", "edited_message", "channel_post", "my_chat_member"):
            chat = (upd.get(key) or {}).get("chat")
            if chat:
                name = chat.get("title") or " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x) or chat.get("username") or "unknown"
                chats[chat["id"]] = f"{name} ({chat.get('type', '?')})"
    return chats, None


def speak_alert(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()