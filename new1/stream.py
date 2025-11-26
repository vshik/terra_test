import requests

API_URL = "http://localhost:8000/chat"

payload = {
    "user_input": user_text,
    "history": st.session_state.history,
    "mcp_logs": st.session_state.mcp_logs,
}

resp = requests.post(API_URL, json=payload).json()

st.write(resp["result"])
st.session_state.history = resp["history"]
st.session_state.mcp_logs = resp["mcp_logs"]
