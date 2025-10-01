#
# Ollama経由でGoogleのGemmaを使用するシンプルなチャットAIプログラム
# コンテクスト、引用、テキストファイルで情報を与え、それに則って質問に答えるが、
# チャットの内容は記憶しない
# 簡単な質問の受け答えに最適
# ask_ai()に例的な使い方を示しています
# また、python3 simple_gemma_ai_ja.py で日本語によるシンプルな受け答えを開始します
# 猫に関する情報がコンテクストとして与えられているので、「我が家でオスの猫は？」などと質問してみてください
#
import sys
import json
from enum import Enum
import ollama
from ollama import Client
import rag

# Class SimpleGemmaAi
class SimpleGemmaAi:
    # Ask
    def ask(
        question: str,
        system_prompt: str = "",
        quote: str = None,
        context: str = None,
        context_data: list[str] = None, # Ex. ["aaa.pdf", "https://example.com", "bbb.doc"]
        response_format: str = "text",
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

        # Append context_data conten
        if bool(context_data):
            ctx = rag.load_data(data=context_data)
            user_content = "Here is additional information:\n" + ctx.combined_ctx
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
    # Set system prompt
    system_prompt = "あなたは優しくて何でも知っています。質問に日本語で答えて下さい。"

    # Get context data
    context = None
    context_data = None
    if len(sys.argv) > 1:
        # Use specified files/URLs as context
        context_data = sys.argv[1:]

        # Note if context_data is too large, 
        # AI will not answer correctly due to its context window size limit 

        # To avoid to read specified files/URLs repeatedly, do following instead
        '''
        ctx = rag.load_data(data=context_data)
        context = "Here is additional information:\n" + ctx.combined_ctx
        context_data = None
        '''
    else:
        # Set original context
        context = "我が家には4匹の猫がいます。猫の名前は「クロ」「ソラ」「ビリー」「シロ」で、シロだけがメスです。\nクロは乱暴で、ソラは食いしん坊で、ビリーはフレンドリー、シロはいつも天井をキョロキョロ眺めています。\nクロは14歳で、ソラは8歳、ビリーは7歳、シロは1歳です。ビリーだけが国外で生まれました。"

    # Ask question
    while True:
        question = ''
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
                    print("\n* Asking AI....")
                    answer = SimpleGemmaAi.ask(
                                system_prompt=system_prompt,
                                context=context,
                                context_data=context_data,
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
