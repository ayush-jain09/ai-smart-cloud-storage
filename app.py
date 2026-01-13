import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from ai.classifier import classify_file
from ai.smart_search import build_index_for_user, search_index
from ai.anomaly_detector import compute_user_risk
from ai.duplicate_detector import sha256_file

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("APP_SECRET", "change-me-in-production")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB
    return app

app = create_app()
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FileObject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    original_name = db.Column(db.String(255), nullable=False)
    stored_name = db.Column(db.String(255), nullable=False)
    mime = db.Column(db.String(120), nullable=True)
    size_bytes = db.Column(db.Integer, nullable=False, default=0)
    sha256 = db.Column(db.String(64), nullable=False, index=True)
    category = db.Column(db.String(32), nullable=False, default="Other")
    tags = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed_at = db.Column(db.DateTime, nullable=True)
    access_count = db.Column(db.Integer, default=0)

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    action = db.Column(db.String(32), nullable=False)  # upload/download/delete/login/search
    file_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()

init_db()

def user_storage_path(user_id: int) -> Path:
    p = STORAGE_DIR / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def log_action(action: str, file_id=None):
    try:
        db.session.add(AccessLog(user_id=current_user.id, action=action, file_id=file_id))
        db.session.commit()
    except Exception:
        db.session.rollback()

def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"

@app.route("/", methods=["GET"])
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            with app.app_context():
                db.session.add(AccessLog(user_id=user.id, action="login", file_id=None))
                db.session.commit()
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("auth/login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        if len(username) < 3:
            flash("Username must be at least 3 characters.", "warning")
            return render_template("auth/register.html")
        if password != password2:
            flash("Passwords do not match.", "warning")
            return render_template("auth/register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "warning")
            return render_template("auth/register.html")
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return render_template("auth/register.html")

        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please sign in.", "success")
        return redirect(url_for("login"))
    return render_template("auth/register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    files = FileObject.query.filter_by(user_id=current_user.id).order_by(FileObject.created_at.desc()).all()
    total_bytes = sum(f.size_bytes for f in files)
    total_files = len(files)

    # Category distribution
    cat_counts = {}
    for f in files:
        cat_counts[f.category] = cat_counts.get(f.category, 0) + 1

    # Activity last 7 days
    start = datetime.utcnow() - timedelta(days=6)
    logs = AccessLog.query.filter(
        AccessLog.user_id == current_user.id,
        AccessLog.created_at >= start
    ).all()

    day_counts = { (start + timedelta(days=i)).strftime("%Y-%m-%d"): 0 for i in range(7) }
    for l in logs:
        k = l.created_at.strftime("%Y-%m-%d")
        if k in day_counts:
            day_counts[k] += 1

    risk = compute_user_risk(logs, files)

    return render_template(
        "dashboard.html",
        total_files=total_files,
        total_storage=human_bytes(total_bytes),
        cat_counts=cat_counts,
        day_counts=day_counts,
        risk=risk,
        recent_files=files[:6]
    )

@app.route("/files", methods=["GET", "POST"])
@login_required
def files_page():
    if request.method == "POST":
        upload = request.files.get("file")
        tags = request.form.get("tags", "").strip()
        if not upload or upload.filename == "":
            flash("Please choose a file to upload.", "warning")
            return redirect(url_for("files_page"))

        filename = secure_filename(upload.filename)
        if not filename:
            flash("Invalid filename.", "danger")
            return redirect(url_for("files_page"))

        # Save temporarily to compute sha and size
        temp_path = user_storage_path(current_user.id) / f"__temp__{filename}"
        upload.save(temp_path)

        file_sha = sha256_file(temp_path)
        size_bytes = temp_path.stat().st_size

        # Duplicate check
        existing = FileObject.query.filter_by(user_id=current_user.id, sha256=file_sha).first()
        if existing:
            temp_path.unlink(missing_ok=True)
            flash(f"Duplicate detected: you already uploaded '{existing.original_name}'.", "info")
            return redirect(url_for("files_page"))

        # AI classification
        mime = upload.mimetype or "application/octet-stream"
        category = classify_file(filename, mime, temp_path)

        stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file_sha[:12]}_{filename}"
        final_path = user_storage_path(current_user.id) / stored_name
        temp_path.replace(final_path)

        obj = FileObject(
            user_id=current_user.id,
            original_name=filename,
            stored_name=stored_name,
            mime=mime,
            size_bytes=size_bytes,
            sha256=file_sha,
            category=category,
            tags=tags
        )
        db.session.add(obj)
        db.session.commit()

        db.session.add(AccessLog(user_id=current_user.id, action="upload", file_id=obj.id))
        db.session.commit()

        # Rebuild search index lazily (small projects). For bigger, do background jobs.
        build_index_for_user(current_user.id, FileObject, user_storage_path(current_user.id))

        flash("Uploaded successfully.", "success")
        return redirect(url_for("files_page"))

    # GET
    q = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()
    tag = request.args.get("tag", "").strip()

    query = FileObject.query.filter_by(user_id=current_user.id)
    if category:
        query = query.filter(FileObject.category == category)
    if tag:
        query = query.filter(FileObject.tags.like(f"%{tag}%"))
    files = query.order_by(FileObject.created_at.desc()).all()

    categories = sorted({f.category for f in FileObject.query.filter_by(user_id=current_user.id).all()})
    return render_template("files.html", files=files, categories=categories, q=q, filter_category=category, filter_tag=tag)

@app.route("/search", methods=["GET"])
@login_required
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return redirect(url_for("files_page"))
    files = FileObject.query.filter_by(user_id=current_user.id).all()
    build_index_for_user(current_user.id, FileObject, user_storage_path(current_user.id))
    ranked_ids = search_index(current_user.id, q)
    ranked = []
    id_to_file = {f.id: f for f in files}
    for fid in ranked_ids:
        if fid in id_to_file:
            ranked.append(id_to_file[fid])

    db.session.add(AccessLog(user_id=current_user.id, action="search", file_id=None))
    db.session.commit()

    flash(f"AI Search results for: {q}", "info")
    categories = sorted({f.category for f in files})
    return render_template("files.html", files=ranked, categories=categories, q=q, filter_category="", filter_tag="")

@app.route("/download/<int:file_id>")
@login_required
def download(file_id):
    f = FileObject.query.filter_by(id=file_id, user_id=current_user.id).first_or_404()
    # update access stats
    f.last_accessed_at = datetime.utcnow()
    f.access_count = (f.access_count or 0) + 1
    db.session.add(f)
    db.session.add(AccessLog(user_id=current_user.id, action="download", file_id=f.id))
    db.session.commit()
    return send_from_directory(user_storage_path(current_user.id), f.stored_name, as_attachment=True, download_name=f.original_name)

@app.route("/delete/<int:file_id>", methods=["POST"])
@login_required
def delete(file_id):
    f = FileObject.query.filter_by(id=file_id, user_id=current_user.id).first_or_404()
    path = user_storage_path(current_user.id) / f.stored_name
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

    db.session.delete(f)
    db.session.add(AccessLog(user_id=current_user.id, action="delete", file_id=file_id))
    db.session.commit()

    build_index_for_user(current_user.id, FileObject, user_storage_path(current_user.id))

    flash("Deleted successfully.", "success")
    return redirect(url_for("files_page"))

@app.route("/insights")
@login_required
def insights():
    files = FileObject.query.filter_by(user_id=current_user.id).all()
    logs = AccessLog.query.filter_by(user_id=current_user.id).order_by(AccessLog.created_at.desc()).limit(200).all()

    # Duplicates are prevented by sha, but we can show "near-duplicate" by same name.
    name_counts = {}
    for f in files:
        name_counts[f.original_name] = name_counts.get(f.original_name, 0) + 1
    near_dupes = [name for name, c in name_counts.items() if c > 1]

    # Storage optimization recommendations
    recs = []
    # Rarely accessed: access_count == 0 older than 14 days
    cutoff = datetime.utcnow() - timedelta(days=14)
    stale = [f for f in files if (f.access_count or 0) == 0 and f.created_at < cutoff]
    if stale:
        recs.append({
            "title": "Archive rarely used files",
            "text": f"{len(stale)} files haven't been accessed in 14+ days. Consider archiving or compressing them."
        })
    big = sorted(files, key=lambda x: x.size_bytes, reverse=True)[:5]
    if big:
        recs.append({
            "title": "Compress large files",
            "text": "Your top large files can be compressed to save storage (especially PDFs, logs, and CSVs)."
        })
    if not recs:
        recs.append({"title": "All good", "text": "No major optimization actions detected. Keep going!"})

    risk = compute_user_risk(logs, files)

    return render_template("insights.html", near_dupes=near_dupes, recs=recs, risk=risk, logs=logs)

@app.template_filter("dt")
def fmt_dt(d):
    if not d:
        return "—"
    return d.strftime("%d %b %Y, %H:%M")

@app.template_filter("bytes")
def fmt_bytes(n):
    return human_bytes(int(n or 0))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
