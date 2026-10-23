"""Streamlit app: upload a painting and predict which painter made it.

Run with:
    streamlit run app.py

Expects the checkpoint saved after training (e.g. "artist_mobilenet_final.pt").
Both formats are supported: a dict with a "model_state" key (plus optional
"classes" and "variant"), or a plain state_dict.

Styling follows the "Deep Impressions" title slide:
  background  #F1EEEA (warm off-white), near-black text
  title       Bodoni (Bodoni Moda from Google Fonts, closest free match to Bodoni FLF)
  subtitle    IM Fell Double Pica Italic
  project     EB Garamond
  small text  Shippori Mincho, letter-spaced
  captions    Cardo (italic)
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import mobilenet_v2
from torchvision.transforms import v2

IMG_SIZE = 224
DEFAULT_CHECKPOINT = "models/model_mobilenet_tuned.pt"
# Alphabetical order, as produced by sorted(train_df["artist"].unique()).
# Only used if the checkpoint does not store its own class list.
DEFAULT_CLASSES = ["Cézanne", "Manet", "Monet", "Morisot", "Renoir", "Van Gogh"]

# Header text (same as the title slide)
TITLE = "DEEP IMPRESSIONS:"
SUBTITLE = "Training a Neural Network to Recognize Painters"
PROJECT = "Project 2 | Machine Learning: From Data to Predictions"
AUTHOR = "Gabriel Gomes"
COURSE = "Ironhack Data Science &amp; Machine Learning Bootcamp"
DATE = "23 Oct 2026"

BG = "#F1EEEA"          # slide background
BG_SOFT = "#E8E4DE"     # slightly darker panels
INK = "#111111"         # text


# ---------------------------------------------------------------- model
class TransferNet(nn.Module):
    """Inference version of the training model: MobileNetV2 features + new head.

    The random augmentation layers used in training are inactive in evaluation
    mode, so they are left out here. They hold no parameters, so the saved
    weights load unchanged. The pretrained ImageNet weights are not downloaded
    because the fine-tuned weights come from the checkpoint.
    """

    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.base = mobilenet_v2(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):                       # x: raw pixels 0-255
        x = x.float() / 255
        x = (x - self.mean) / self.std
        return self.head(self.pool(self.base(x)))


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    """Load the checkpoint once and keep the model in memory between reruns."""
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        # Checkpoints saved with extra Python objects may need weights_only=False.
        # Only do this for files you created yourself.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    is_wrapped = isinstance(ckpt, dict) and "model_state" in ckpt
    state = ckpt["model_state"] if is_wrapped else ckpt
    classes = ckpt.get("classes", DEFAULT_CLASSES) if is_wrapped else DEFAULT_CLASSES
    variant = ckpt.get("variant", "unknown") if is_wrapped else "unknown"

    model = TransferNet(num_classes=len(classes))
    model.load_state_dict(state)
    model.eval()
    return model, list(classes), variant


# ---------------------------------------------------------------- inference
# Same preprocessing as in training/evaluation: resize the shorter side,
# center-crop to a square, keep raw 0-255 values (the model scales them).
PREPROCESS = v2.Compose([
    v2.Resize(IMG_SIZE),
    v2.CenterCrop(IMG_SIZE),
    v2.PILToTensor(),
])


def predict(model, image: Image.Image) -> torch.Tensor:
    x = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        return F.softmax(model(x), dim=1).squeeze(0)


# ---------------------------------------------------------------- style
STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:wght@400&family=Cardo:ital,wght@0,400;1,400&family=EB+Garamond&family=IM+Fell+Double+Pica:ital@0;1&family=Shippori+Mincho&display=swap');

:root {{
  --bg: {BG};
  --bg-soft: {BG_SOFT};
  --ink: {INK};
  --font-title: 'Bodoni Moda', 'Bodoni 72', 'Didot', Georgia, serif;
  --font-italic: 'IM Fell Double Pica', 'Cardo', Georgia, serif;
  --font-project: 'EB Garamond', Garamond, Georgia, serif;
  --font-small: 'Shippori Mincho', 'EB Garamond', Georgia, serif;
  --font-caption: 'Cardo', Georgia, serif;
}}

/* page background and base text */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background-color: var(--bg) !important;
  color: var(--ink) !important;
  color-scheme: light;
}}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
  background-color: var(--bg-soft) !important;
}}
.block-container {{ padding-top: 3rem; max-width: 1200px; }}
#MainMenu, footer {{ visibility: hidden; }}

.stApp, .stMarkdown p, .stMarkdown li, [data-testid="stText"],
[data-testid="stAlert"] p, [data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
  font-family: var(--font-project) !important;
  color: var(--ink) !important;
  font-size: 1.1rem;
}}

/* header, as on the title slide */
h1.deep {{
  font-family: var(--font-title) !important;
  font-weight: 400 !important;
  font-size: clamp(2.4rem, 5.2vw, 4.8rem) !important;
  line-height: 1.02 !important;
  letter-spacing: 0.01em;
  color: var(--ink) !important;
  margin: 0 !important;
  padding: 0 !important;
}}
p.subtitle {{
  font-family: var(--font-italic) !important;
  font-style: italic;
  font-size: clamp(1.4rem, 2.6vw, 2.3rem) !important;
  line-height: 1.15;
  margin: 0.3rem 0 0.7rem 0 !important;
}}
p.project {{
  font-family: var(--font-project) !important;
  font-size: 1.2rem !important;
  letter-spacing: 0.02em;
  margin: 0 !important;
}}
.meta {{ text-align: right; }}
.meta p.author {{
  font-family: var(--font-italic) !important;
  font-style: italic;
  font-size: 1.6rem !important;
  letter-spacing: 0.06em;
  margin: 0 0 1.1rem 0 !important;
}}
.meta p.small {{
  font-family: var(--font-small) !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.08em;
  line-height: 1.35;
  margin: 0 0 0.6rem 0 !important;
}}
hr.rule {{ border: 0; border-top: 1px solid rgba(17,17,17,0.35); margin: 1.6rem 0 1.4rem 0; }}

/* section headings */
h2.section {{
  font-family: var(--font-title) !important;
  font-weight: 400 !important;
  font-size: 1.9rem !important;
  margin: 0 0 0.2rem 0 !important;
  padding: 0 !important;
}}
p.lede {{
  font-family: var(--font-italic) !important;
  font-style: italic;
  font-size: 1.25rem !important;
  margin: 0 0 1rem 0 !important;
}}

/* prediction card */
.result-label {{
  font-family: var(--font-small); font-size: 0.85rem; letter-spacing: 0.14em;
  text-transform: uppercase; margin: 0 0 0.2rem 0;
}}
.result-name {{
  font-family: var(--font-title); font-size: clamp(2.2rem, 4.2vw, 3.6rem);
  line-height: 1.05; margin: 0;
}}
.result-conf {{
  font-family: var(--font-italic); font-style: italic; font-size: 1.5rem; margin: 0.2rem 0 0.6rem 0;
}}
.result-note {{
  font-family: var(--font-project); font-size: 1.1rem; border-left: 2px solid var(--ink);
  padding-left: 0.8rem; margin: 0.4rem 0 0.8rem 0;
}}

/* widgets */
label, [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  font-family: var(--font-small) !important;
  letter-spacing: 0.06em;
  color: var(--ink) !important;
}}
[data-testid="stSidebar"] h2 {{ font-family: var(--font-title) !important; font-weight: 400; letter-spacing: 0; }}
[data-testid="stFileUploaderDropzone"] {{
  background-color: var(--bg-soft) !important;
  border: 1px dashed rgba(17,17,17,0.55) !important;
}}
[data-testid="stFileUploaderDropzone"] button, .stButton button {{
  background-color: transparent !important; color: var(--ink) !important;
  border: 1px solid var(--ink) !important; font-family: var(--font-small) !important;
  letter-spacing: 0.06em;
}}
[data-testid="stTextInput"] input {{
  background-color: var(--bg) !important; color: var(--ink) !important;
  border: 1px solid rgba(17,17,17,0.45) !important; font-family: var(--font-caption) !important;
}}
[data-testid="stAlert"] {{ background-color: var(--bg-soft) !important; border: 1px solid rgba(17,17,17,0.3); }}
[data-testid="stExpander"] {{ border: 1px solid rgba(17,17,17,0.3) !important; background: transparent !important; }}
[data-testid="stExpander"] summary p {{ font-family: var(--font-small) !important; letter-spacing: 0.06em; }}

/* image captions: Cardo italic, right-aligned like the slide's credit line */
[data-testid="stImageCaption"], [data-testid="stImageCaption"] p {{
  font-family: var(--font-caption) !important; font-style: italic; text-align: right;
  font-size: 1rem !important; color: var(--ink) !important;
}}
[data-testid="stDataFrame"] {{ font-family: var(--font-caption) !important; }}
</style>
"""


def render_header() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    left, right = st.columns([3, 1.3])
    with left:
        st.markdown(
            f'<h1 class="deep">{TITLE}</h1>'
            f'<p class="subtitle">{SUBTITLE}</p>'
            f'<p class="project">{PROJECT}</p>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="meta"><p class="author">{AUTHOR}</p>'
            f'<p class="small">{COURSE}</p>',#<p class="small">{DATE}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr class="rule">', unsafe_allow_html=True)


# ---------------------------------------------------------------- interface
st.set_page_config(page_title="Deep Impressions: Who painted this?", page_icon="🎨", layout="wide")
render_header()

st.markdown(
    '<h2 class="section">Who painted this?</h2>'
    '<p class="project">Upload a painting and the model will suggest which of six painters made it: '
    'Cézanne, Manet, Monet, Morisot, Renoir or Van Gogh.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    checkpoint = st.text_input("Model checkpoint", DEFAULT_CHECKPOINT)

if not Path(checkpoint).is_file():
    st.error(f"Checkpoint not found: `{checkpoint}`. Put the file next to `app.py` "
             "or enter its full path in the sidebar.")
    st.stop()

model, classes, variant = load_model(checkpoint)
st.sidebar.caption(f"Model variant: {variant}")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload a painting to get a prediction.")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

probs = predict(model, image)
order = probs.argsort(descending=True)
top1 = classes[order[0]]
conf1 = probs[order[0]].item()

left, right = st.columns([1, 1], gap="large")

with left:
    st.image(image, caption=uploaded.name, use_container_width=True)

with right:
    st.markdown(
        f'<p class="result-label">Predicted painter</p>'
        f'<p class="result-name">{top1}</p>'
        f'<p class="result-conf">{conf1:.0%} confidence</p>',
        unsafe_allow_html=True,
    )

    chart = (
        pd.DataFrame({"painter": classes, "probability": probs.tolist()})
        .sort_values("probability", ascending=False)
        .set_index("painter")
    )
    st.bar_chart(chart, horizontal=True, color=INK)
    st.dataframe(chart.style.format("{:.1%}"), use_container_width=True)

with st.expander("How to read this result"):
    st.markdown(
        """
- The model is a **closed-set classifier**: it always picks one of the six painters,
  even for a painting by someone else (for example Degas, Pissarro or Sisley).
- It was trained on reproductions from WikiArt. Photos of framed paintings, sketches,
  drawings and heavy crops are outside what it has seen, so expect lower accuracy.
- It is **overconfident**: wrong predictions often still show 65-75% confidence.
- Manet and Morisot are the weakest classes (small training sets and varied styles),
  so a Manet prediction in particular should be checked by an expert.
- This is an assistive tool for first-pass suggestions, not a way to authenticate a work.
        """
    )
