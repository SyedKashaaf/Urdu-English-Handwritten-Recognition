
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import json, cv2
import numpy as np

# ── Model architecture ─────────────────────────────────────
class SharedEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.1),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.1),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.project = nn.Sequential(
            nn.Flatten(),nn.Linear(256,feature_dim),
            nn.LayerNorm(feature_dim),nn.ReLU(),nn.Dropout(0.3),
        )
    def forward(self, x): return self.project(self.cnn(x))

class CharHead(nn.Module):
    def __init__(self, feature_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim,128),nn.ReLU(),
            nn.Dropout(0.3),nn.Linear(128,n_classes))
    def forward(self, x): return self.net(x)

class LangHead(nn.Module):
    def __init__(self, feature_dim, n_langs=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim,256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,n_langs))
    def forward(self, x): return self.net(x)

class HWRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        fd = config["feature_dim"]
        self.encoder    = SharedEncoder(fd)
        self.feature_dim = fd
        self.char_heads = nn.ModuleDict({
            "urdu_chars"   : CharHead(fd, len(config["urdu_classes"])),
            "english_chars": CharHead(fd, len(config["english_classes"])),
        })
        self.lang_head = LangHead(fd, 3)
    def encode(self, x): return self.encoder(x)
    def classify(self, x, lang):
        return self.char_heads[lang](self.encode(x))
    def detect_language(self, x):
        return self.lang_head(self.encode(x))

# ── Load model ─────────────────────────────────────────────
DEVICE = "cpu"
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)
model = HWRModel(config)
ckpt  = torch.load("model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model loaded — epoch {ckpt['epoch']} acc={ckpt['val_acc']:.4f}")

URDU_CLASSES    = config["urdu_classes"]
ENGLISH_CLASSES = config["english_classes"]
LANG_THRESHOLD  = 0.65

# ══════════════════════════════════════════════════════════
# PREPROCESSING — matches training exactly
# Training used Otsu binarization + white background
# ══════════════════════════════════════════════════════════
def preprocess_for_model(pil_image):
    """Convert PIL image to tensor matching training preprocessing."""
    img = np.array(pil_image.convert("L"))
    # Otsu binarization — CRITICAL: matches training
    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure white background (dark character on white)
    if img.mean() < 127:
        img = cv2.bitwise_not(img)
    # Resize to 64x64
    img = cv2.resize(img, (64, 64))
    # To tensor and normalize
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)  # (1,1,64,64)

def count_characters(pil_image):
    """
    Count connected components to detect single vs multiple chars.
    Returns number of significant character-like regions.
    """
    img = np.array(pil_image.convert("L"))
    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove tiny noise
    kernel  = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Count components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    h, w = img.shape
    min_area = h * w * 0.005  # at least 0.5% of image
    max_area = h * w * 0.95   # not the whole image
    significant = sum(
        1 for i in range(1, num_labels)
        if min_area < stats[i, cv2.CC_STAT_AREA] < max_area
    )
    return significant

def detect_cjk(pil_image):
    """Detect CJK (Chinese/Japanese/Korean) characters by stroke analysis."""
    img = np.array(pil_image.convert("L"))
    h, w = img.shape
    # CJK characters have very high stroke density
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    density = binary.sum() / (h * w * 255)
    # Count grid cells with strokes (CJK fills space uniformly)
    grid = 4
    cell_h, cell_w = h//grid, w//grid
    cells_with_strokes = 0
    for i in range(grid):
        for j in range(grid):
            cell = binary[i*cell_h:(i+1)*cell_h,
                          j*cell_w:(j+1)*cell_w]
            if cell.sum() / (cell_h * cell_w * 255) > 0.02:
                cells_with_strokes += 1
    # CJK full page has strokes in many grid cells
    return cells_with_strokes >= 12 and density > 0.08

# ── Helper: confidence bar HTML ────────────────────────────
def prob_bar(label, pct, color):
    w = min(int(pct), 100)
    return f"""
    <div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;
            margin-bottom:3px;font-size:13px">
            <span>{label}</span>
            <span style="font-weight:600">{pct:.1f}%</span>
        </div>
        <div style="background:#E0E0E0;border-radius:99px;height:7px">
            <div style="background:{color};width:{w}%;
                height:7px;border-radius:99px"></div>
        </div>
    </div>"""

def warning_html(msg, detail=""):
    return f"""
    <div style="font-family:sans-serif;padding:16px">
        <div style="background:#FFF8E1;border-left:4px solid #FFC107;
            border-radius:8px;padding:16px">
            <div style="font-size:20px;font-weight:700;color:#F57F17">
                ⚠️ {msg}</div>
            <div style="color:#666;margin-top:8px;font-size:13px;
                line-height:1.5">{detail}</div>
        </div>
    </div>"""

def unknown_html(lang_probs, reason=""):
    u = lang_probs[0].item()*100
    e = lang_probs[1].item()*100
    k = lang_probs[2].item()*100
    return f"""
    <div style="font-family:sans-serif;padding:16px">
        <div style="background:#FFF3F3;border-left:4px solid #F44336;
            border-radius:8px;padding:16px;margin-bottom:16px">
            <div style="font-size:22px;font-weight:700;color:#D32F2F">
                ❓ Unknown Language</div>
            <div style="color:#666;margin-top:6px;font-size:13px">
                {reason if reason else
                 "This character does not belong to Urdu or English."}
            </div>
        </div>
        <div style="background:#F9F9F9;border-radius:8px;padding:14px">
            <div style="font-size:12px;font-weight:600;color:#444;
                margin-bottom:10px">Language Probabilities</div>
            {prob_bar("🇵🇰 Urdu", u, "#1565C0")}
            {prob_bar("🇬🇧 English", e, "#2E7D32")}
            {prob_bar("❓ Unknown", k, "#C62828")}
        </div>
    </div>"""

# ── Main inference ─────────────────────────────────────────
@torch.no_grad()
def predict_image(image):
    if image is None:
        return """<div style="padding:40px;text-align:center;
            color:#999;font-family:sans-serif">
            <div style="font-size:48px">📂</div>
            <div style="margin-top:10px">
                Upload a single handwritten character
            </div></div>"""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # ── Step 1: Check if image has multiple characters ──
    n_chars = count_characters(image)
    if n_chars > 5:
        return warning_html(
            "Multiple Characters Detected",
            f"Found ~{n_chars} character regions in this image.<br><br>"
            "This model recognizes <b>one character at a time</b>.<br>"
            "Please upload an image with a <b>single isolated character</b>.<br><br>"
            "Examples: one Urdu letter, one English letter, one digit.")

    # ── Step 2: Check for CJK (Chinese/Japanese/Korean) ──
    if detect_cjk(image):
        from torch import zeros
        fake_probs = zeros(3)
        fake_probs[2] = 1.0
        return unknown_html(
            type("p", (), {"item": lambda self: 0.0})(),
            "Detected: Chinese / Japanese / Korean script.<br>"
            "This model supports Urdu and English only.")

    # ── Step 3: Preprocess matching training ───────────
    x = preprocess_for_model(image)

    # ── Step 4: Language detection ─────────────────────
    lang_logits = model.detect_language(x)
    lang_probs  = F.softmax(lang_logits, dim=-1)[0]
    lang_idx    = lang_probs.argmax().item()
    lang_conf   = lang_probs[lang_idx].item()

    # ── Step 5: Unknown threshold check ────────────────
    if lang_conf < LANG_THRESHOLD or lang_idx == 2:
        return unknown_html(lang_probs)

    # ── Step 6: Character recognition ──────────────────
    lang_key = "urdu_chars" if lang_idx==0 else "english_chars"
    char_logits = model.classify(x, lang_key)
    char_probs  = F.softmax(char_logits, dim=-1)[0]
    top3_v, top3_i = char_probs.topk(3)
    classes = URDU_CLASSES if lang_idx==0 else ENGLISH_CLASSES
    top3    = [(classes[i.item()], v.item())
               for i, v in zip(top3_i, top3_v)]

    flag  = "🇵🇰" if lang_idx==0 else "🇬🇧"
    lang  = "Urdu (اردو)" if lang_idx==0 else "English"
    color = "#1565C0" if lang_idx==0 else "#2E7D32"
    bg    = "#EFF6FF" if lang_idx==0 else "#F0FFF4"
    char  = top3[0][0]
    cconf = top3[0][1]*100
    lconf = lang_conf*100

    medals = ["🥇","🥈","🥉"]
    top3_html = ""
    for i,(ch,p) in enumerate(top3):
        w = int(p*100)
        top3_html += f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:3px">
                <span style="font-size:14px;font-weight:500">
                    {medals[i]} {ch}</span>
                <span style="font-size:13px;color:#555;
                    font-weight:600">{p*100:.1f}%</span>
            </div>
            <div style="background:#E0E0E0;border-radius:99px;height:7px">
                <div style="background:{color};
                    width:{min(w,100)}%;
                    height:7px;border-radius:99px"></div>
            </div>
        </div>"""

    return f"""
    <div style="font-family:sans-serif;padding:16px">
        <div style="background:{bg};border-left:4px solid {color};
            border-radius:8px;padding:16px;margin-bottom:14px">
            <div style="font-size:12px;color:#666;font-weight:500;
                text-transform:uppercase;letter-spacing:.05em;
                margin-bottom:3px">Detected Language</div>
            <div style="font-size:24px;font-weight:700;color:{color}">
                {flag} {lang}</div>
            <div style="margin-top:8px">
                <span style="background:{color};color:white;
                    padding:2px 12px;border-radius:99px;font-size:12px">
                    {lconf:.1f}% confident</span>
            </div>
        </div>
        <div style="background:#F9F9F9;border-radius:8px;
            padding:16px;margin-bottom:14px;text-align:center">
            <div style="font-size:12px;color:#666;font-weight:500;
                text-transform:uppercase;letter-spacing:.05em;
                margin-bottom:8px">Recognized Character</div>
            <div style="font-size:56px;font-weight:700;
                color:#1A1A1A;line-height:1.2">{char}</div>
            <div style="margin-top:10px">
                <span style="background:#4CAF50;color:white;
                    padding:3px 14px;border-radius:99px;font-size:13px">
                    ✓ {cconf:.1f}% confidence</span>
            </div>
        </div>
        <div style="background:#F9F9F9;border-radius:8px;padding:14px">
            <div style="font-size:12px;color:#666;font-weight:600;
                text-transform:uppercase;letter-spacing:.05em;
                margin-bottom:12px">Top 3 Predictions</div>
            {top3_html}
        </div>
    </div>"""

# ── Text inference ─────────────────────────────────────────
def predict_text(text):
    if not text or not text.strip():
        return """<div style="padding:40px;text-align:center;
            color:#999;font-family:sans-serif">
            <div style="font-size:48px">🌐</div>
            <div style="margin-top:10px">Type text to detect language
            </div></div>"""

    urdu_c = sum(1 for c in text if "\u0600"<=c<="\u06FF")
    eng_c  = sum(1 for c in text if c.isascii() and c.isalpha())
    cjk_c  = sum(1 for c in text if "\u4e00"<=c<="\u9fff")
    total  = max(urdu_c+eng_c, 1)

    if cjk_c > 2:
        lang,conf,color,flag,script = (
            "Unknown","99","#C62828","❓","CJK Script")
    elif urdu_c/total > 0.6:
        lang,conf,color,flag,script = (
            "Urdu (اردو)",f"{urdu_c/total*100:.0f}",
            "#1565C0","🇵🇰","Arabic/Nastaliq — Right to Left")
    elif eng_c/total > 0.6:
        lang,conf,color,flag,script = (
            "English",f"{eng_c/total*100:.0f}",
            "#2E7D32","🇬🇧","Latin — Left to Right")
    else:
        lang,conf,color,flag,script = (
            "Unknown","50","#C62828","❓",
            "Mixed or unrecognized script")

    words = len(text.split())
    chars = len(text.replace(" ",""))
    direction = ("RTL" if "Right" in script
                 else "LTR" if "Left" in script else "N/A")
    bg = ("#EFF6FF" if "Urdu" in lang
          else "#F0FFF4" if "English" in lang
          else "#FFF3F3")

    return f"""
    <div style="font-family:sans-serif;padding:16px">
        <div style="background:{bg};border-left:4px solid {color};
            border-radius:8px;padding:16px;margin-bottom:14px">
            <div style="font-size:12px;color:#666;font-weight:500;
                text-transform:uppercase;letter-spacing:.05em;
                margin-bottom:3px">Detected Language</div>
            <div style="font-size:24px;font-weight:700;color:{color}">
                {flag} {lang}</div>
            <div style="margin-top:8px">
                <span style="background:{color};color:white;
                    padding:2px 12px;border-radius:99px;font-size:12px">
                    {conf}% confident</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
            gap:10px;margin-bottom:14px">
            <div style="background:#EFF6FF;border-radius:8px;
                padding:14px;text-align:center">
                <div style="font-size:28px;font-weight:700;
                    color:#1565C0">{words}</div>
                <div style="font-size:11px;color:#666;margin-top:3px">
                    Words</div>
            </div>
            <div style="background:#F0FFF4;border-radius:8px;
                padding:14px;text-align:center">
                <div style="font-size:28px;font-weight:700;
                    color:#2E7D32">{chars}</div>
                <div style="font-size:11px;color:#666;margin-top:3px">
                    Characters</div>
            </div>
            <div style="background:#FFFBEB;border-radius:8px;
                padding:14px;text-align:center">
                <div style="font-size:22px;font-weight:700;
                    color:#D97706">{direction}</div>
                <div style="font-size:11px;color:#666;margin-top:3px">
                    Direction</div>
            </div>
        </div>
        <div style="background:#F8F8F8;border-radius:8px;padding:12px">
            <div style="font-size:11px;color:#888;
                text-transform:uppercase;letter-spacing:.05em">Script</div>
            <div style="font-size:14px;font-weight:500;color:#333;
                margin-top:3px">{script}</div>
        </div>
    </div>"""

# ── Gradio UI ──────────────────────────────────────────────
css = """
.gradio-container{max-width:880px !important;margin:auto !important}
footer{display:none !important}
"""

with gr.Blocks(
    title="Urdu & English Handwriting Recognition",
    theme=gr.themes.Soft(primary_hue="blue",
                         secondary_hue="green"),
    css=css,
) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 12px 0">
        <div style="font-size:40px">✍️</div>
        <h1 style="margin:8px 0 4px;font-size:26px;
            font-weight:700;color:#111">
            Multilingual Handwriting Recognition
        </h1>
        <p style="color:#666;margin:0 0 6px;font-size:14px">
            Recognizes <b>Urdu (اردو)</b> and <b>English</b>
            handwritten characters and digits
        </p>
        <div style="background:#FFF8E1;border-radius:8px;
            padding:8px 16px;display:inline-block;
            margin-bottom:12px;font-size:12px;color:#92400E">
            ⚠️ Upload <b>ONE character at a time</b>
            — not full pages or paragraphs
        </div>
        <div style="display:flex;justify-content:center;
            gap:8px;flex-wrap:wrap">
            <span style="background:#EFF6FF;color:#1565C0;
                padding:4px 14px;border-radius:99px;
                font-size:12px;font-weight:600">
                🇵🇰 Urdu 88.3%</span>
            <span style="background:#F0FFF4;color:#2E7D32;
                padding:4px 14px;border-radius:99px;
                font-size:12px;font-weight:600">
                🇬🇧 English 97.9%</span>
            <span style="background:#F3F4F6;color:#374151;
                padding:4px 14px;border-radius:99px;
                font-size:12px;font-weight:600">
                🎯 Lang Detection 99.6%</span>
            <span style="background:#FFF1F2;color:#BE123C;
                padding:4px 14px;border-radius:99px;
                font-size:12px;font-weight:600">
                🚫 Unknown Rejection 99.7%</span>
        </div>
    </div>""")

    with gr.Tabs():

        with gr.Tab("📷  Handwritten Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(
                        label="Upload Single Character",
                        type="pil", height=260)
                    img_btn = gr.Button(
                        "🔍  Recognize Character",
                        variant="primary", size="lg")
                    gr.HTML("""
                    <div style="background:#FFFBEB;
                        border-radius:8px;padding:12px 14px;
                        margin-top:8px;font-size:12px;
                        color:#78350F;line-height:1.6">
                        <b>✅ Good inputs:</b><br>
                        • One Urdu character (ب، ج، ک...)<br>
                        • One English letter (A, b, Z...)<br>
                        • One digit (0-9)<br><br>
                        <b>❌ Bad inputs:</b><br>
                        • Full pages or paragraphs<br>
                        • Multiple characters at once<br>
                        • Chinese, Hindi, Arabic pages
                    </div>""")
                with gr.Column(scale=1):
                    img_out = gr.HTML(
                        value="""<div style="height:320px;
                        display:flex;align-items:center;
                        justify-content:center;color:#999;
                        font-family:sans-serif;text-align:center">
                        <div><div style="font-size:48px">🔍</div>
                        <div style="margin-top:8px;font-size:14px">
                        Upload one character and click Recognize
                        </div></div></div>""")
            img_btn.click(fn=predict_image,
                         inputs=[img_in], outputs=[img_out])

        with gr.Tab("⌨️  Type Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt_in = gr.Textbox(
                        label="Enter Urdu or English Text",
                        placeholder=(
                            "Type here...\n"
                            "e.g. Hello world\n"
                            "یا اردو میں لکھیں"),
                        lines=5)
                    txt_btn = gr.Button(
                        "🌐  Detect Language",
                        variant="primary", size="lg")
                    gr.Examples(
                        label="Try these examples",
                        examples=[
                            ["Hello my name is Ahmed"],
                            ["Machine learning is amazing"],
                            ["آپ کا نام کیا ہے"],
                            ["پاکستان زندہ باد"],
                            ["میں AI پڑھتا ہوں"],
                        ], inputs=txt_in)
                with gr.Column(scale=1):
                    txt_out = gr.HTML(
                        value="""<div style="height:200px;
                        display:flex;align-items:center;
                        justify-content:center;color:#999;
                        font-family:sans-serif;text-align:center">
                        <div><div style="font-size:48px">🌐</div>
                        <div style="margin-top:8px">
                        Type text and click Detect
                        </div></div></div>""")
            txt_btn.click(fn=predict_text,
                         inputs=[txt_in], outputs=[txt_out])

        with gr.Tab("ℹ️  About"):
            gr.HTML("""
            <div style="font-family:sans-serif;
                max-width:620px;margin:auto;padding:20px">
                <h2>About This Project</h2>
                <p style="color:#555;font-size:14px">
                    ML course project — BS Artificial Intelligence.
                    Recognizes single handwritten characters and digits
                    in Urdu and English.
                </p>
                <h3>⚠️ Important Limitation</h3>
                <div style="background:#FFF8E1;border-radius:8px;
                    padding:12px;font-size:13px;color:#78350F">
                    This model is trained on <b>single isolated
                    characters (64×64 pixels)</b>.<br>
                    It cannot read full words, lines or paragraphs.
                    Upload one character at a time for best results.
                </div>
                <h3 style="margin-top:16px">ML Techniques</h3>
                <div style="display:grid;
                    grid-template-columns:1fr 1fr;gap:10px">
                    <div style="background:#EFF6FF;border-radius:8px;
                        padding:12px">
                        <b style="color:#1E40AF">CNN</b>
                        <div style="color:#3B82F6;font-size:12px;
                            margin-top:2px">Shared encoder</div>
                    </div>
                    <div style="background:#F0FFF4;border-radius:8px;
                        padding:12px">
                        <b style="color:#166534">GAN</b>
                        <div style="color:#16A34A;font-size:12px;
                            margin-top:2px">Synthetic data</div>
                    </div>
                    <div style="background:#FDF4FF;border-radius:8px;
                        padding:12px">
                        <b style="color:#7E22CE">Semi-Supervised</b>
                        <div style="color:#9333EA;font-size:12px;
                            margin-top:2px">Pseudo-labeling</div>
                    </div>
                    <div style="background:#FFFBEB;border-radius:8px;
                        padding:12px">
                        <b style="color:#92400E">Transfer Learning</b>
                        <div style="color:#D97706;font-size:12px;
                            margin-top:2px">Add languages easily</div>
                    </div>
                </div>
                <h3 style="margin-top:16px">Results</h3>
                <p style="color:#555;font-size:13px">
                    Urdu: 88.3% | English: 97.9% |
                    Lang Detection: 99.6% | Unknown Rejection: 99.7%
                </p>
                <h3>Author</h3>
                <p style="color:#555;font-size:14px">
                    <b>Hasnain Sherazi</b> —
                    BS AI, Machine Learning Project
                </p>
            </div>""")

demo.launch()
