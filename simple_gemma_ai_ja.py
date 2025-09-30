#
# Ollama経由でGoogleのGemmaを使用するシンプルなチャットAIプログラム
# コンテクスト、引用、テキストファイルで情報を与え、それに則って質問に答えるが、
# チャットの内容は記憶しない
# 簡単な質問の受け答えに最適
# ask_ai()に例的な使い方を示しています
# また、python3 simple_gemma_ai_ja.py で日本語によるシンプルな受け答えを開始します
# 猫に関する情報がコンテクストとして与えられているので、「我が家でオスの猫は？」などと質問してみてください
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
        system_prompt: str = "",
        quote: str = None,
        context: str = None,
        response_format: str = "text",
        file: str = None, # Currently supports text file only
        ollama_host: str = "localhost", # Ollama host running LLM
        ollama_port: int = 11434,
        model = "gemma3n:e4b", # LLM model: "gemma3:latest", "gemma3:12b", ....
        temperature = 0.0, 
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

# Ask AI
def ask_ai():
    # Ask question
    while True:
        question = ''
        res = None
        terminate = False
        print("\n- 質問を入力してください。複数行でもOKです。\n" +
              "- 質問入力後、Enterを押し、Ctrl-Dを押すとAIが質問に答えます。\n" +
              "- 何も入力しないでCtrl-Dを押すか、'exit'とだけ入力してEnterを押すと終了します。\n")

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
                    print("\n* AIに問い合わせ中....")
                    answer = SimpleGemmaAi.ask(
                                system_prompt="あなたは優しくて何でも知っています。質問に日本語で答えて下さい。",
                                context="我が家には4匹の猫がいます。猫の名前は「クロ」「ソラ」「ビリー」「シロ」で、シロだけがメスです。\nクロは乱暴で、ソラは食いしん坊で、ビリーはフレンドリー、シロはいつも天井をキョロキョロ眺めています。\nクロは14歳で、ソラは8歳、ビリーは7歳、シロは1歳です。",
                                question=question,
                                ollama_host="localhost",
                                model="gemma3n:e4b",)

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
        print('\n' + answer.rstrip('\n') + '\n')

# Main 
if __name__ == "__main__":
    ask_ai()
