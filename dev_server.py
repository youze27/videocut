"""Local dev server for UI smoke-testing the real dashboard without the AI
stack (torch/opencv/ffmpeg). Uses the real dashboard blueprint + templates and
points SNAPSHOTS_DIR/CLIPS_DIR at ./devtest. Not part of the Docker image."""
import os
from pathlib import Path

BASE = str(Path(__file__).resolve().parent / "devtest")
os.environ["OUTPUT_DIR"] = os.path.join(BASE, "output")
os.environ["SNAPSHOTS_DIR"] = os.path.join(BASE, "snapshots")
os.environ["CLIPS_DIR"] = os.path.join(BASE, "clips")
os.environ["WORK_DIR"] = os.path.join(BASE, "work")

from flask import Flask, render_template
from ui.web_server import bp

app = Flask(__name__, template_folder="ui/templates")
app.register_blueprint(bp)
app.add_url_rule("/config", "config_page", lambda: render_template("config.html"))

if __name__ == "__main__":
    print("dev dashboard -> http://127.0.0.1:8080/  (config at /config)")
    app.run(host="0.0.0.0", port=8080, debug=False)
