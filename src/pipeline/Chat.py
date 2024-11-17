import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

class Saint:
    def __init__(self,api_key: str):
        self.model = ChatGroq(model_name="llama3-8b-8192",api_key=api_key)
        self.store = {}
        self.output_parser = StrOutputParser()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieve the chat history for a given session ID, or create a new one."""
        return self.store.setdefault(session_id, ChatMessageHistory())

    def setup_chain(self):
        prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a compassionate guide, deeply inspired by the wisdom of Lord Krishna.
         Respond to the user's questions with insights that reflect Krishna's teachings, adapting your tone to suit their emotional state.
         If the user expresses distress or longing, offer comforting and empathetic advice.
         Encourage reflection by asking thoughtful, related questions.
         Be concise unless a detailed response is necessary to address the query."""
        
        ),MessagesPlaceholder(variable_name="thinking")
        ])
        return prompt | self.model

    def chat(self,user_input: str):
        """Main chat loop to handle user input and generate responses."""

        chain = self.setup_chain()
        with_message_history = RunnableWithMessageHistory(chain, self.get_session_history)
        
        while True:
            user_input = user_input
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"saint: Goodbye,! Have a nice day.")
                break

            config = {"configurable": {"session_id": None}}
            
            output_parser = StrOutputParser()
            chain = with_message_history | output_parser
            final_output = chain.invoke([HumanMessage(content=user_input)], config=config)
            
            # print(f"you: {user_input}")
            # print(f"saint: {final_output}\n")
            return final_output

if __name__ == "__main__":
    bot = Saint()
    bot.chat()