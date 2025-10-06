import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# -------------------------
# Load Model + Mappings
# -------------------------
MODEL_PATH = "mobilenet_fashion_best.h5"
MAPPINGS_PATH = "label_mappings.json"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load mappings
with open(MAPPINGS_PATH, "r") as f:
    label_mappings = json.load(f)

idx_to_article = {int(k): v for k, v in label_mappings["article"].items()}
idx_to_color   = {int(k): v for k, v in label_mappings["color"].items()}
idx_to_usage   = {int(k): v for k, v in label_mappings["usage"].items()}

# -------------------------
# Preprocess function
# -------------------------
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# -------------------------
# Prediction function (with conditions)
# -------------------------
def predict(img):
    x = preprocess(img)
    preds = model.predict(x)

    # --- Article Type ---
    article_probs = preds[0][0]
    article_idx = np.argmax(article_probs)
    article_label = idx_to_article[article_idx]
    article_conf = article_probs[article_idx] * 100

    # --- BaseColour ---
    color_probs = preds[1][0]
    top2_idx = np.argsort(color_probs)[-2:][::-1]
    top1, top2 = top2_idx[0], top2_idx[1]
    top1_label, top1_conf = idx_to_color[top1], color_probs[top1] * 100
    top2_label, top2_conf = idx_to_color[top2], color_probs[top2] * 100

    if top1_conf < 30:
        final_colors = [("Unreadable", top1_conf)]
    elif top1_conf < 40:
        final_colors = [(top1_label, top1_conf)]
    elif top1_conf >= 40 and (top1_conf - top2_conf) <= 15:
        final_colors = [(top1_label, top1_conf), (top2_label, top2_conf)]
    else:
        final_colors = [(top1_label, top1_conf)]

    # --- Usage ---
    usage_probs = preds[2][0]
    usage_idx = np.argmax(usage_probs)
    usage_label = idx_to_usage[usage_idx]
    usage_conf = usage_probs[usage_idx] * 100

    return (article_label, article_conf), final_colors, (usage_label, usage_conf)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart Catalog", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .card {
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        background-color: #f9f9f9;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    .stProgress > div > div {
        border-radius: 10px;
    }
    h1 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("👕 Quick Catalog - Clothing Recognition")
st.write("Upload or capture an image of clothing. The system will auto-recognize **Article Type, Color, and Usage**.")

# Layout: two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Upload / Capture")
    uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a picture")

    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
    elif camera_file is not None:
        img = Image.open(camera_file).convert("RGB")

    if img is not None:
        st.image(img, caption="Selected Image", use_container_width=True)

with col2:
    st.header("📊 Predictions")

    if img is not None:
        # Run prediction
        article, colors, usage = predict(img)

        # --- Display results as cards ---
        st.markdown("#### 🔎 Recognition Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👕 Article")
            st.success(f"{article[0]} \n({article[1]:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎨 Colour(s)")
            for c, p in colors:
                st.write(f"{c}: {p:.1f}%")
                st.progress(int(p))
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏷️ Usage")
            st.info(f"{usage[0]} \n({usage[1]:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # --- Editable dropdowns ---
        st.markdown("### ✏️ Confirm & Save")
        with st.container():
            article_choice = st.selectbox(
                "Article Type",
                list(idx_to_article.values()),
                index=list(idx_to_article.values()).index(article[0])
            )

            predicted_colors = [] if colors[0][0] == "Unreadable" else [c for c, _ in colors]
            color_choice = st.multiselect(
                "Base Colour(s)",
                list(idx_to_color.values()),
                default=predicted_colors
            )

            usage_choice = st.selectbox(
                "Usage",
                list(idx_to_usage.values()),
                index=list(idx_to_usage.values()).index(usage[0])
            )

            if st.button("✅ Save"):
                st.success(f"Saved: {article_choice}, {color_choice}, {usage_choice}")
    else:
        st.info("👈 Please upload or capture an image to see predictions.")
