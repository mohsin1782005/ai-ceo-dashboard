import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 1. Page Setup
st.set_page_config(page_title="My AI Data Dashboard", layout="wide")
st.title("📊 The CEO's AI Dashboard")

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Configuration Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Option 1: Check if the key is in Streamlit Secrets (Hidden for users)
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API Key loaded securely from the cloud.")
    
    # Option 2: If no secret is found, ask the user to enter it
    else:
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        if not api_key:
            st.warning("⚠️ Please enter an API Key to proceed.")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# 4. Main App Logic
if uploaded_file is not None and api_key:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize the AI
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )

    # Create the Agent
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        allow_dangerous_code=True 
    )

    # 5. Chat Input
    user_question = st.chat_input("Ask your data a question...")

    if user_question:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # MASTER INSTRUCTION: Force the AI to save files instead of just talking
                enhanced_question = user_question + """
                IMPORTANT INSTRUCTIONS:
                1. If I ask for a TABLE or LIST (e.g., "Show me top 5..."), save the result as a CSV file named 'table.csv'.
                2. If I ask for a CHART, save the figure as 'chart.png'.
                3. Do NOT try to show the table in text format. Just say "Here is the data."
                """
                
                response = agent.invoke(enhanced_question)
                st.markdown(response['output'])
                st.session_state.messages.append({"role": "assistant", "content": response['output']})

                # CHECK: Did the agent generate a Chart?
                if os.path.exists("chart.png"):
                    st.image("chart.png", caption="Generated Chart")
                    os.remove("chart.png")

                # CHECK: Did the agent generate a Table?
                if os.path.exists("table.csv"):
                    st.write("### 📋 Generated Data")
                    # Load the new CSV and display it interactively
                    result_df = pd.read_csv("table.csv")
                    st.dataframe(result_df)
                    os.remove("table.csv")

elif uploaded_file is not None and not api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
else:

    st.info("Please upload a CSV file to begin.")
