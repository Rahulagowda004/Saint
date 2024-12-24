link: https://saint-bot.streamlit.app/

---

# 🌟 **Saint - Your Spiritual Guide**

Saint is a Streamlit-based application powered by LangChain and Groq AI, offering spiritual guidance inspired by the teachings of Lord Krishna. With a compassionate and empathetic tone, the chatbot helps users find peace, clarity, and wisdom in their life's journey.

---

## 🛠️ **Features**

- **Spiritual Guidance**: Get answers inspired by Lord Krishna's teachings.
- **Emotionally Adaptive Responses**: The chatbot adapts its tone based on the user's emotional state.
- **Reflection Encouragement**: Thoughtful, concise answers encourage users to introspect and grow.
- **Secure Access**: Uses Groq API Key for interaction with the LLM (Llama3-8b-8192 model).
- **Streamlit Sidebar**: Easily enter your Groq API Key for a personalized experience.

---

## 🚀 **Getting Started**

### Prerequisites
- Python 3.10+
- A Groq API key. [Get yours here](https://console.groq.com/keys)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/saint-spiritual-guide.git
   cd saint-spiritual-guide
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Run the Application
   ```bash
   streamlit run app.py
   ```

---

## 🖼️ **Uploading Application Pictures**
To make your repository more visually appealing and helpful:
1. Take screenshots of the application interface:
   - **Homepage**: Highlight the title, caption, and Groq API input section in the sidebar.
   - **Chat Interaction**: Show examples of user messages and chatbot responses.
   - **Sidebar Input**: Focus on how users can enter their Groq API Key.
2. Save these screenshots as:
   - `homepage.png`
   - `chat_interaction.png`
   - `sidebar_input.png`
3. Create a `screenshots` folder in your repository and add these images.
4. Add the following section to the README:

### Screenshots
| **Homepage**                          | **Chat Interaction**                      | **Sidebar Input**                         |
|---------------------------------------|------------------------------------------|------------------------------------------|
| ![Homepage](screenshots/homepage.png) | ![Chat Interaction](screenshots/chat_interaction.png) | ![Sidebar Input](screenshots/sidebar_input.png) |

---

## 📜 **How It Works**
1. The user enters their Groq API Key in the Streamlit sidebar.
2. The chatbot leverages the Llama3-8b-8192 model via LangChain to process queries.
3. Responses are concise, compassionate, and inspired by the teachings of Lord Krishna.
4. Session history is maintained to provide contextually aware answers during the chat.

---

## 🤝 **Contributing**
We welcome contributions! Feel free to fork the repository and submit a pull request. Ensure your code adheres to best practices and includes meaningful comments.

---

## 📄 **License**
This project is licensed under the MIT License.

---

By including screenshots, users will better understand the application and its interface, making your project more engaging and accessible.
