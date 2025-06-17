import streamlit as st
import json
import os
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime

# File to store history
HISTORY_FILE = "chat_history.json"

# Function to load history from file
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Function to save history to file
def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file)

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "response" not in st.session_state:
    st.session_state.response = ""
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""
if "show_viewport" not in st.session_state:
    st.session_state.show_viewport = False
if "selected_timestamp" not in st.session_state:
    st.session_state.selected_timestamp = ""
if "history" not in st.session_state:
    st.session_state.history = load_history()

# Streamlit UI
st.title('🐍 Python AI Assistant')
st.header('Provide your Python code and get your doubts clarified here.')
st.subheader("🔮 Powered with LLaMA 3", divider='rainbow')

# User input
st.session_state.input_text = st.text_area("💬 Ask your questions here:", value=st.session_state.input_text,)

# Submit button
if st.button("🚀 Submit") and st.session_state.input_text:
    # Store the current input before processing
    current_input = st.session_state.input_text
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.selected_question = current_input
    st.session_state.selected_timestamp = current_time
    st.session_state.show_viewport = True
    
    with st.spinner("🤖 AI is thinking..."):
        # Define LLM and output parser
        llm = Ollama(model="llama3")
        output_parser = StrOutputParser()

        # Define a basic prompt message 
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're a Python expert, and your task is to help users with clean, simple, and reliable Python code. When explaining code,Clarity: Make the response beginner-friendly. Break down complex concepts into simple, digestible explanations.Security: Always prioritize security. For any code dealing with user data, include proper security measures such as password hashing (e.g., using bcrypt or werkzeug.security), secure cookies, and input validation.Completeness: Include all the necessary components for a full implementation. When applicable, explain how to set up a proper database, handle user authentication, and validate input safely. Include error handling to ensure the code works as expected in different scenarios.Code Quality: Follow best practices for code quality: write clean, modular code, and include relevant comments explaining what each part of the code does. Ensure the code can scale to more complex applications.Edge Cases: Consider potential edge cases and errors users might encounter. Make sure the code can handle incorrect inputs, missing data, and other common issues.Relevance: Stay on topic, and make sure your code and explanations are highly relevant to the question. If you suggest improvements or advanced features, explain why they are necessary and how they enhance the solution.Performance: Ensure the code is optimized for performance, particularly for common tasks in web development, such as handling form submissions, working with databases, and managing sessions.When you provide examples, always recommend production-ready practices and include any additional relevant considerations, such as database setup or dealing with large datasets."),
                ("user", "Question:{question}")
            ]
        )
        
        # Chain: prompt -> model -> output parser
        chain = prompt | llm | output_parser

        # Get the result
        st.session_state.response = chain.invoke({"question": current_input})

        # Save the response to history with timestamp
        st.session_state.history.append({
            "question": current_input,
            "response": st.session_state.response,
            "timestamp": current_time
        })
        save_history(st.session_state.history)  # Save history to file
        
        # Only clear the input after successful processing
        st.session_state.input_text = ""

# Chat history in sidebar
with st.sidebar:
    st.title("📜 Chat History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            timestamp = entry.get("timestamp", "No time")
            with st.expander(f"💭 {entry['question'][:30]}...", expanded=False):
                st.write(f"**Time:** {timestamp}")
                st.write("**Question:**")
                st.write(entry['question'])
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("View", key=f"view_{i}"):
                        st.session_state.response = entry['response']
                        st.session_state.selected_question = entry['question']
                        st.session_state.selected_timestamp = timestamp
                        st.session_state.show_viewport = True
                with col_b:
                    if st.button("Delete", key=f"delete_{i}"):
                        idx = len(st.session_state.history) - 1 - i
                        if idx >= 0 and idx < len(st.session_state.history):
                            st.session_state.history.pop(idx)
                            save_history(st.session_state.history)
                            if st.session_state.show_viewport:
                                st.session_state.show_viewport = False
                            st.rerun()

# Display response viewport in main area
if st.session_state.show_viewport and st.session_state.response:
    st.markdown("---")  # Add separator
    with st.container():
        st.markdown("### 🤖 Response Viewer")
        st.info(f"**Question:** {st.session_state.selected_question}")
        st.markdown("#### Response:")
        st.write(st.session_state.response)
        st.caption(f"*Timestamp: {st.session_state.selected_timestamp}*")
        if st.button("Close"):
            st.session_state.show_viewport = False
            st.rerun()