from pathlib import Path

CODE_HINTS = ["import ", "def ", "class ", "console.log", "#include", "public static", "SELECT ", "CREATE TABLE", "function(", "=>"]
DOC_EXT = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs", ".php", ".html", ".css", ".sql", ".ipynb"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
MEDIA_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv", ".avi"}
TEXT_EXT = {".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml", ".xml"}

def classify_file(filename: str, mime: str, path: Path) -> str:
    ext = Path(filename).suffix.lower()

    if ext in IMAGE_EXT or mime.startswith("image/"):
        return "Image"
    if ext in MEDIA_EXT or mime.startswith("audio/") or mime.startswith("video/"):
        return "Media"
    if ext in CODE_EXT:
        return "Code"
    if ext in DOC_EXT:
        return "Document"
    if ext in TEXT_EXT or mime.startswith("text/"):
        # Heuristic: check if looks like code
        try:
            sample = path.read_text(errors="ignore")[:4000]
            s = sample.replace("\r", "")
            if any(h in s for h in CODE_HINTS):
                return "Code"
            if len(s.strip()) == 0:
                return "Other"
            return "Text"
        except Exception:
            return "Text"
    return "Other"
