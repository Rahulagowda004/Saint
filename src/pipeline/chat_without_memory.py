import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
import sys

class Saint:
    def __init__(self,api_key: str):
        os.environ['GROQ_API_KEY'] = api_key
        self.llm = ChatGroq(model_name="llama3-8b-8192")
        self.output_parser = StrOutputParser()
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
            """You are a compassionate guide, deeply inspired by the wisdom of Lord Krishna.
                Respond to the user's questions with insights that reflect Krishna's teachings, adapting your tone to suit their emotional state.
                If the user expresses distress or longing, offer comforting and empathetic advice.
                Encourage reflection by asking thoughtful, related questions.
                Be concise unless a detailed response is necessary to address the query."""
            ),
            HumanMessagePromptTemplate.from_template(
                "Question: {question}\nAnswer:"
            )])
        self.chain = self.chat_prompt | self.llm | self.output_parser
        
    def chat(self,question: str):
        return self.chain.invoke({"question": question})
        
        
if __name__ == "__main__":
    saint = Saint(api_key=os.getenv('GROQ_API_KEY'))
    print(saint.chat("I,m not fine"))

