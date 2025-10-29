import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Load base model dan adapter
BASE_MODEL = "microsoft/Phi-4-mini-reasoning"
ADAPTER = "dtp-fine-tuning/phi4-alpacaid"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# Streamlit UI
st.title("💬 Chatbot Phi-4 AlpacaID")
st.caption("Model fine-tuning untuk instruksi dalam Bahasa Indonesia")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Tulis pertanyaanmu di sini...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Format prompt
    prompt = "### Instruction:\n" + user_input + "\n\n### Response:\n"

    response = generator(
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )[0]["generated_text"]

    # Ambil hanya bagian respons
    response_clean = response.split("### Response:\n")[-1].strip()
    st.session_state.chat_history.append({"role": "assistant", "content": response_clean})

# Tampilkan riwayat chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])