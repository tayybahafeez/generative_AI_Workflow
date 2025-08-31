import streamlit as st
from rag_model import rag_model
from speech_to_text import transcribe_from_mic, transcribe_from_file

st.set_page_config(page_title="Voice AI Assistant", layout="centered")
st.title("🎙️ Voice/Text AI Assistant for Magnetismo")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("#### 💬 Enter your question (type or speak):")

# === Text Input ===
user_input = st.text_input("Type your question:")

# === Voice Input Options ===
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Speak (Mic Input)"):
        try:
            user_input = transcribe_from_mic()
            st.success(f"Transcribed: {user_input}")
        except Exception as e:
            st.error(f"Microphone Error: {e}")

with col2:
    uploaded_file = st.file_uploader("📁 Upload audio file", type=["mp3", "wav", "m4a"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            user_input = transcribe_from_file(tmp_path)
            st.success(f"Transcribed: {user_input}")
        except Exception as e:
            st.error(f"File Transcription Error: {e}")

# === Process Input ===
if user_input and st.button("💡 Get Answer"):
    answer = rag_model.get_answer(user_input)
    st.session_state.chat_history.append((user_input, answer))

# === Display Chat ===
for user_q, bot_a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {user_q}")
    st.markdown(f"**🤖 Assistant:** {bot_a}")
    st.markdown("---")

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
