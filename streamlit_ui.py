import streamlit as st
import requests
import json

API_URL = "https://ai-financial-advisor-9sdc.onrender.com/ask"

st.title("💰 FinGenie")

# 1) Collect user profile once
if "profile" not in st.session_state:
    with st.form("profile_form"):
        age = st.number_input("Age", 18, 80, 28)
        income = st.number_input("Monthly Income (₹)", 10000, 200000, 60000)
        risk = st.selectbox("Risk Appetite", ["Low", "Moderate", "High"])
        goals = st.multiselect("Goals", ["Insurance", "Invest", "Retirement"])
        submitted = st.form_submit_button("Save & Continue")
        if submitted:
            st.session_state.profile = {
                "age": age, "income": income * 12,
                "risk": risk, "goals": goals
            }
            st.success("Profile saved!")

# 2) Chat loop
if "profile" in st.session_state:
    query = st.chat_input("Ask me about term plans, MFs, loans …")
    if query:
        with st.spinner("Thinking…"):
            payload = {"profile": st.session_state.profile, "query": query}
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                st.write("🧑‍💼 **Advisor:**", data["answer"])
                # if "sources" in data:
                #     if st.checkbox("Show sources"):
                #         for s in data["sources"]:
                #             st.caption(f'• {s}')
            else:
                st.error("Error: Unable to fetch response.")

# 3) Export chat
if st.button("Export Chat"):
    chat_log = st.session_state.get("chat", [])
    st.download_button("Download", json.dumps(chat_log), file_name="chat_log.json")
