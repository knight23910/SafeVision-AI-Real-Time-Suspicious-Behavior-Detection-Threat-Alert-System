import sys, os
from functools import wraps
from flask import Flask, request, send_file, render_template_string, session, redirect, url_for

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.db_utils import get_incidents, mark_false_alarm
import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
app.secret_key = "change-this-secret-key"

STYLE = """
<style>
  :root {
    --bg:#0b0c14; --panel:#12131f; --panel2:#171826; --border:#242539;
    --text:#e8e9f3; --muted:#8b8da3; --accent:#5b7cff;
    --critical:#ff4d5e; --high:#ff9a4d; --medium:#ffcf4d; --low:#3ddc84;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,'Segoe UI',Roboto,sans-serif; margin:0; }
  .shell { max-width:1200px; margin:0 auto; padding:28px 24px; }
  .navbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:28px; }
  .brand { display:flex; align-items:center; gap:10px; font-size:20px; font-weight:700; }
  .brand-dot { width:10px; height:10px; border-radius:50%; background:var(--critical); box-shadow:0 0 10px var(--critical); }
  .user-chip { display:flex; align-items:center; gap:12px; background:var(--panel); padding:8px 14px; border-radius:30px; border:1px solid var(--border); font-size:13px; }
  .role-badge { background:var(--accent); color:#fff; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600; text-transform:uppercase; }
  a.link { color:var(--muted); text-decoration:none; font-size:13px; }
  a.link:hover { color:var(--text); }

  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }
  .stat-card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px 20px; }
  .stat-num { font-size:28px; font-weight:700; }
  .stat-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.5px; margin-top:4px; }
  .stat-card.critical .stat-num { color:var(--critical); }
  .stat-card.high .stat-num { color:var(--high); }
  .stat-card.medium .stat-num { color:var(--medium); }

  .panel { background:var(--panel); border:1px solid var(--border); border-radius:16px; overflow:hidden; }
  .panel-head { display:flex; justify-content:space-between; align-items:center; padding:18px 22px; border-bottom:1px solid var(--border); }
  .panel-head h2 { margin:0; font-size:16px; }
  select { background:var(--panel2); color:var(--text); border:1px solid var(--border); padding:8px 14px; border-radius:8px; font-size:13px; }

  table { width:100%; border-collapse:collapse; }
  th { text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.5px; padding:12px 22px; border-bottom:1px solid var(--border); }
  td { padding:14px 22px; border-bottom:1px solid var(--border); font-size:14px; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:var(--panel2); }

  .badge { padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; text-transform:uppercase; display:inline-block; }
  .badge.Critical { background:rgba(255,77,94,0.15); color:var(--critical); }
  .badge.High { background:rgba(255,154,77,0.15); color:var(--high); }
  .badge.Medium { background:rgba(255,207,77,0.15); color:var(--medium); }
  .badge.Low { background:rgba(61,220,132,0.15); color:var(--low); }

  .btn { background:var(--panel2); color:var(--text); border:1px solid var(--border); padding:6px 12px; border-radius:8px; font-size:12px; cursor:pointer; margin-right:6px; text-decoration:none; display:inline-block; }
  .btn:hover { border-color:var(--accent); color:var(--accent); }

  .modal-bg { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); align-items:center; justify-content:center; z-index:99; }
  .modal-bg.show { display:flex; }
  .modal-box { background:var(--panel); border-radius:16px; padding:16px; border:1px solid var(--border); }
  .modal-box video { max-width:70vw; max-height:70vh; border-radius:10px; display:block; }
  .close-btn { text-align:right; margin-bottom:8px; }

  .login-wrap { display:flex; align-items:center; justify-content:center; height:100vh; }
  .login-box { background:var(--panel); border:1px solid var(--border); padding:36px; border-radius:16px; width:300px; }
  .login-box input { width:100%; padding:10px 12px; margin-bottom:12px; background:var(--panel2); border:1px solid var(--border); border-radius:8px; color:var(--text); }
  .login-box button { width:100%; padding:10px; background:var(--accent); border:none; border-radius:8px; color:#fff; font-weight:600; cursor:pointer; }
</style>
"""

LOGIN_PAGE = STYLE + """
<body>
  <div class="login-wrap">
    <form method="post" class="login-box">
      <div class="brand" style="margin-bottom:20px;"><span class="brand-dot"></span> Threat Detection</div>
      {% if error %}<p style="color:var(--critical); font-size:13px;">{{error}}</p>{% endif %}
      <input name="username" placeholder="Username">
      <input name="password" type="password" placeholder="Password">
      <button type="submit">Log in</button>
    </form>
  </div>
</body>
"""

PAGE = STYLE + """
<body>
  <div class="shell">
    <div class="navbar">
      <div class="brand"><span class="brand-dot"></span> Threat Detection Dashboard</div>
      <div class="user-chip">{{user}} <span class="role-badge">{{role}}</span> <a class="link" href="/logout">Log out</a></div>
    </div>

    <div class="stats">
      <div class="stat-card"><div class="stat-num">{{total}}</div><div class="stat-label">Total Incidents</div></div>
      <div class="stat-card critical"><div class="stat-num">{{critical_count}}</div><div class="stat-label">Critical</div></div>
      <div class="stat-card high"><div class="stat-num">{{high_count}}</div><div class="stat-label">High</div></div>
      <div class="stat-card medium"><div class="stat-num">{{medium_count}}</div><div class="stat-label">Medium</div></div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <h2>Recent Incidents</h2>
        <form method="get">
          <select name="level" onchange="this.form.submit()">
            <option value="">All levels</option>
            {% for lvl in ["Critical","High","Medium","Low"] %}
              <option value="{{lvl}}" {% if level==lvl %}selected{% endif %}>{{lvl}}</option>
            {% endfor %}
          </select>
        </form>
      </div>
      <table>
        <tr><th>Camera</th><th>Time</th><th>Level</th><th>Risk</th><th>Category</th><th>Behavior</th><th>Scene</th><th>Evidence</th><th>Feedback</th></tr>
        {% for row in incidents %}
        <tr>
          <td>Cam {{row[1]}}</td><td>{{row[2]}}</td>
          <td><span class="badge {{row[3]}}">{{row[3]}}</span></td>
          <td>{{row[4]}}</td><td>{{row[5]}}</td><td>{{row[6]}}</td><td>{{row[7]}}</td>
          <td>
            <a class="btn" href="javascript:void(0)" onclick="playVid('/evidence/{{row[9].split('/')[-1]}}')">Blurred</a>
            {% if role=="admin" %}<a class="btn" href="javascript:void(0)" onclick="playVid('/evidence-raw/{{row[8].split('/')[-1]}}')">Raw</a>{% endif %}
          </td>
          <td>
            {% if row[10] %}<span style="color:var(--muted);">Marked false alarm</span>
            {% else %}<a class="btn" href="/false-alarm/{{row[0]}}" onclick="return confirm('Mark this as a false alarm? This helps improve future accuracy.')">Not a threat</a>{% endif %}
          </td>
        </tr>
        {% endfor %}
      </table>
    </div>
  </div>

  <div class="modal-bg" id="modal">
    <div class="modal-box">
      <div class="close-btn"><a class="link" href="javascript:void(0)" onclick="closeVid()">✕ Close</a></div>
      <video id="v" controls></video>
    </div>
  </div>
  <script>
    function playVid(src){ document.getElementById('v').src=src; document.getElementById('modal').classList.add('show'); }
    function closeVid(){ document.getElementById('modal').classList.remove('show'); document.getElementById('v').pause(); }
  </script>
</body>
"""


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "role" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form.get("username"), request.form.get("password")
        user = config.USERS.get(u)
        if user and user["password"] == p:
            session["username"], session["role"] = u, user["role"]
            return redirect(url_for("dashboard"))
        return redirect(url_for("login", error="1"))  # fresh GET reload, not a stale POST-rendered page
    error = "Invalid username or password" if request.args.get("error") else None
    return render_template_string(LOGIN_PAGE, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    level = request.args.get("level") or None
    incidents = get_incidents(level=level)
    all_incidents = get_incidents()
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for row in all_incidents:
        counts[row[3]] = counts.get(row[3], 0) + 1
    return render_template_string(PAGE, incidents=incidents, level=level, user=session["username"], role=session["role"],
                                   total=len(all_incidents), critical_count=counts["Critical"],
                                   high_count=counts["High"], medium_count=counts["Medium"])


@app.route("/false-alarm/<int:incident_id>")
@login_required
def false_alarm(incident_id):
    mark_false_alarm(incident_id)
    return redirect(url_for("dashboard"))


@app.route("/evidence/<filename>")
@login_required
def evidence(filename):
    return send_file(os.path.join(ROOT, "evidence", "blurred", filename))


@app.route("/evidence-raw/<filename>")
@login_required
def evidence_raw(filename):
    if session.get("role") != "admin":
        return "Not authorized", 403
    return send_file(os.path.join(ROOT, "evidence", "raw", filename))


if __name__ == "__main__":
    app.run(debug=True, port=5000)