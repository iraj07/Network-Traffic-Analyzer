import streamlit as st
import joblib
from file_module import run_file_analysis
from live_module import run_live_sniffing

# ۱. تنظیمات سیستمی و ظاهری
st.set_page_config(
    page_title="Guardian AI v5.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ۲. بهینه‌سازی بارگذاری مدل و انکودر
@st.cache_resource(show_spinner=False)
def load_assets():
    try:
        model = joblib.load('models/trained_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
        return model, le
    except Exception:
        return None, None


# ۳. طراحی منوی کناری (Sidebar)
with st.sidebar:
    with st.sidebar:
        # استفاده از آبی برای عنوان جهت ایجاد حس اعتماد
        st.markdown("<h1 style='text-align: center; color: #58A6FF;'>🛡️ GUARDIAN AI</h1>", unsafe_allow_html=True)

        # استفاده از خاکستری روشن برای متن‌های فرعی (ایجاد تضاد بصری)
        st.markdown("<p style='text-align: center; color: #8B949E;'>Next-Gen Intrusion Detection</p>",
                    unsafe_allow_html=True)

        # رنگ دکمه‌ها و وضعیت‌ها به صورت خودکار توسط تم Streamlit مدیریت می‌شود
    st.divider()

    app_mode = st.selectbox(
        "🛠️ SELECT ENGINE:",
        ["📂 File Intelligence", "📡 Live Packet Sniffing"]
    )

    st.divider()

    # نمایش وضعیت سیستم
    model, le = load_assets()
    if model and le:
        st.success("✅ AI Engine: ACTIVE")
        st.caption("Model Version: 5.0.1 (Neural)")
    else:
        st.error("❌ AI Engine: OFFLINE")
        st.warning("Check 'models/' folder for PKL files.")

    # --- اصلاح خطا: جایگزین کردن spacer با فضای خالی ---
    for _ in range(10):
        st.write("")

    st.info("System Health: Stable")

# ۴. مدیریت اجرای بخش‌های مختلف
if app_mode == "📂 File Intelligence":
    if model and le:
        run_file_analysis(model, le)
    else:
        st.error("Model assets not found! Please check the models directory.")
else:
    run_live_sniffing()

# ۵. فوتر
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Guardian AI Security Lab")