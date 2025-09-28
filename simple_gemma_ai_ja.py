#
# GoogleのGemmaを使用したシンプルなチャットAIプログラム
# コンテクスト、引用、テキストファイルで情報を与え、それに則って質問に答えるが、
# 過去のチャットの内容は記憶しない
#
import json
from enum import Enum
import ollama
from ollama import Client

# Class SimpleGemmaAi
class SimpleGemmaAi:
    # Ask
    def ask(
        question: str,
        response_format: str = "text",
        system_prompt: str = "",
        quote: str = None,
        context: str = None,
        file: str = None, # Currently supports text file only
        model = "gemma3n:e4b", # LLM model
                               # "gemma3:latest", "gemma3:12b", ....
        temperature = 0.0, 
        ollama_host: str = "localhost", # Ollama host running LLM
        ollama_port: int = 11434,
    ) -> str:
        # Initialize parameters
        ollama_base_url = f"http://{ollama_host}:{ollama_port}"
        messages = []
        user_content = ""

        # Initialize chat client
        client = Client(host=ollama_base_url,)

        # Set system prompt
        if system_prompt != "":
            messages.append({"role": "system", "content": system_prompt})

        # Include quote
        if bool(quote):
            user_content += "========\n" + quote + "========\n\n"

        # Include context
        if bool(context):
            user_content += context + "\n\n"

        if bool(user_content):
            user_content = "Here is given information:\n" + user_content
            messages.append({"role": "system", "content": user_content})

        # Append file content
        if bool(file):
            with open(file, "r", encoding="utf-8") as file:
                file_content = file.read()
                user_content = "Here is additional information:\n" + file_content
                messages.append({"role": "system", "content": user_content})

        # Set quesiotn
        user_question = "The following is my question:\n" + question
        messages.append({"role": "user", "content": user_question})
        #print(messages)

        # Ask
        response = client.chat(
            model=model,
            messages=messages,
            keep_alive="1h",
            format="" if response_format == "text" else "json",
            options={"temperature": temperature},
        )
        #print(response)

        return response.message.content

# Ask AI question
def ask_ai():
    # Ask question
    while True:
        question = ''
        res = None
        terminate = False
        print("\n- Ask a question. Multiple lines are accepted.\n" +
              "- Press Enter then Ctrl-D to submit. " +
              "Just Ctrl-D or enter 'exit' to terminate.:\n")

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
                    answer = SimpleGemmaAi.ask(
                                system_prompt="あなたは優しくて何でも知っています。質問に日本語で答えて下さい。",
                                context="我が家には4匹の猫がいます。猫の名前は「クロ」「ソラ」「ビリー」「シロ」で、シロだけがメスです。\nクロは乱暴で、ソラは食いしん坊で、ビリーはフレンドリー、シロはいつも天井をキョロキョロ眺めています。\nクロは14歳で、ソラは8歳、ビリーは7歳、シロは1歳です。",
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
        print('\n' + answer)

# Main 
if __name__ == "__main__":
    ask_ai()
