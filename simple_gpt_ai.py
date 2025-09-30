#
# Simple AI chat program using OpenAI GPT
# Can answer questions from given context, quote, a text file but does not remember past chat content
# Best for simple Q&A
# See example usage of SimpleGptAi in ask_ai()
# Also, 'python3 simple_gpt_ai.py' starts a simple Q&A session
# Ask a question like "List all male cats in my house."
#

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger

# Global parameters
DEFAULT_MODEL = "gpt-4o-mini" # Best cost performance
#DEFAULT_MODEL = "gpt-4.1-mini" # Better than gpt-4o-mini but more expensive
#DEFAULT_MODEL = "gpt-4.1-nano" # Not as smart as gpt-4o-mini

# Load OpenAI key, etc
if not load_dotenv():
    print('Error: ".env" is not found or not accessible')
    sys.exit(1)

# Check OPENAI_API_KEY
if os.getenv("OPENAI_API_KEY") is None:
    print('Error: OPENAI_API_KEY is not set')
    sys.exit(1)

# SimpleGptAi class
class SimpleGptAi:
    # Ask
    def ask(
            question: str,
            system_prompt: str = None,
            quote: str = None,
            context: str = None,
            response_format: str = "text",
            file: str = None, # Currently supports text file only
            model: str = DEFAULT_MODEL,
    ) -> str:
        # Intialize
        llm = ChatOpenAI(model=model)
        messages = []
        user_content = ""

        # Include quote
        if bool(quote):
            user_content += "========\n" + quote + "========\n\n"

        # Include context
        if bool(context):
            user_content += context + "\n\n"

        if bool(user_content):
            user_content = "Here is given information:\n" + user_content

       # Append file content
        if bool(file):
            with open(file, "r", encoding="utf-8") as file:
                file_content = file.read()
                user_content += "\n\nHere is additional information:\n" + file_content

        # Set question
        if bool(user_content):
            messages = [
                    ("system", system_prompt),
                    ("human",
                        f"Please answer questions based on the following context:\n{context}\n\nQeustion: {question}"
                    ),
            ]
        else:
            # No context
            messages = [
                    ("system", system_prompt),
                    ("human", question),
            ]

        # Ask
        answer = llm.invoke(messages)

        return answer.content

# Ask question
def ask_ai():
    # Ask question
    while True:
        question = ''
        terminate = False
        print("\n- Ask a question. Multiple lines are accepted.\n- Press Enter then Ctrl-D to submit. Just Ctrl-D or enter 'exit' to terminate.:\n")

        # Accept multiple lines including empty lines
        # Ctrl-D to ask
        while True:
            try:
                # Get input
                q = input()
            except EOFError: # Ctrl-D (submit question to AI)
                # Check len of question
                if len(question) == 0: # Len of question is 0
                    terminate = True
                else:
                    # Ask question to AI
                    print("\n* Asking AI....")
                    answer = SimpleGptAi.ask(
                                system_prompt="You are a very kind assistant. Please answer questions.",
                                context = "There are four cats in my house. The oldest is Kuro, and he is 14 years old. Sora is an 8-year-old sweet cat, and Billy was born in 2018. Shiro is the only female and was born in 2024. Billy is the only cat from outside the US.",
                                question=question)

                break

            # Append input to quesion
            question += q + '\n'

            # Check if question is just 'exit'
            if question == 'exit\n':
                terminate = True
                break

        if terminate:
            exit()

        # Print answer
        print(answer + '\n')

# Main
if __name__ == '__main__':
    ask_ai()
