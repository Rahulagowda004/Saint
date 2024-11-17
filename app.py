import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st

# Load environment variables
load_dotenv()

# Streamlit Sidebar for OpenAI API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", key="chatbot_api_key", type="password")
    "[Get an Groq API key](https://console.groq.com/keys)"
    "[Github Profile](https://github.com/Rahulagowda004?tab=repositories)"
# Streamlit Title
st.title("🌟 **Saint** - Your Spiritual Guide")
st.caption("🕉️ **Tap into the divine wisdom of Lord Krishna** to find peace, clarity, and guidance in every step of your journey. 💫 Let's reflect, learn, and grow together! 🙏")

# Initialize LangChain model
os.environ['GROQ_API_KEY'] = groq_api_key
if not os.environ['GROQ_API_KEY']:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# Setup model and prompt
model = ChatGroq(model_name="llama3-8b-8192", api_key=os.environ['GROQ_API_KEY'])
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a compassionate guide, deeply inspired by the wisdom of Lord Krishna.
    Respond to the user's questions with insights that reflect Krishna's teachings, adapting your tone to suit their emotional state.
    Be concise with answers, do not exceed three lines.
    If the user expresses distress or longing, offer comforting and empathetic advice.
    Encourage reflection by asking thoughtful, related questions.
    """
    ),
    MessagesPlaceholder(variable_name="thinking")
])

chain = prompt | model

# Function to initialize or retrieve session history
def get_session_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = ChatMessageHistory()
    return st.session_state["chat_history"]

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "🌸 **Greetings, dear soul.** How may I assist you on your journey of wisdom and peace today? 🙏"}]

# Display chat messages from session state
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input handling
if user_input := st.chat_input("Enter your message"):
    chat_history = get_session_history()

    # Append user message to session state and history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with_message_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=lambda: chat_history  # Callable returning the chat history
    )

    # Run chain and get the response
    chain_with_parser = with_message_history | output_parser
    response = chain_with_parser.invoke(
        [HumanMessage(content=user_input)],
        config={"configurable": {"session_id": "default_session"}}
    )

    # Append assistant message to session state and update chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
