# chat_ai

A powerful Chat AI with RAG, chat history, and context awareness.  Uses Gemma (via Ollama) and GPT LLM models.  Loads data from TXT, PDF, DOC, CSV, web pages (HTML), spreadsheets (XLS), presentations (PPTX), and LibreOffice documents (ODT).  Offers vectorization, summarization, and other helpful features.

Note: This README is mostly generated with the programs in this repository.


## Prerequisites

\*gemma\*.py programs rely on the following components:

* **Ollama:** A tool for running large language models locally. Ensure Ollama is installed and running on your system (defaults to localhost).
* **Gemini Model:** The program uses the `gemma3n:e4b` language model. This model must be downloaded and accessible within Ollama.
* **Embeddings Model:** The program utilizes the `bge-m3:latest` model for generating embeddings. This model also needs to be downloaded and be available in Ollama.

**Installation:**

1. **Install Ollama:** Follow the instructions on the [Ollama website](https://ollama.com/) to install Ollama for your operating system.
2. **Pull Models:** Ensure the following models are pulled using Ollama:

   ```bash
   ollama pull gemma3n:e4b
   ollama pull bge-m3:latest
   ```

## gemma_ai.py
AI Chat Program using Google Gemma via Ollama

This program leverages the Google Gemma language model, accessed through Ollama, to provide an AI chat experience.

**Key Features:**

*   **Question Answering with RAG:** Can answer questions by retrieving information from provided documents using Retrieval-Augmented Generation (RAG).
*   **Memory:** Remembers past chat content for more contextual conversations.
*   **General Q&A and Chatting:** Suitable for a wide range of general question answering and conversational tasks.

**RAG Data Sources:**

The program supports various data formats for RAG, including:

*   TXT
*   PDF
*   DOC
*   CSV
*   WEB (http)
*   HTML
*   XLS
*   PPTX
*   ODT (LibreOffice)

**Example Usage:**

```python
rag_data = ["aaa.pdf", "https://example.com", "bbb.doc"]
llm = GemmaAi(rag_data=rag_data)
```

**Explanation of Example:**

This example demonstrates how to use the program with RAG. It reads the content from "aaa.pdf", accesses the content of the website "https://example.com", and reads "bbb.doc". The content from these sources is then used to inform the Gemma model's responses.

**Exmaple usage:**
   ```bash
   python3 gemma_ai.py cats.txt
   ```

This reads cats.txt and answers questions based on its content. Ask questions like "Who is the oldest cat in my house and how old is he?"

You can also specify a URL:
   ```bash
   python3 gemma_ai.py https://science.nasa.gov/solar-system/comets/3i-atlas/
   ```
Ask "What is 3I/ATLAS?"


## simple_gemma_ai.py
Simple AI Chat Program using Google Gemma via Ollama

This program provides a simple conversational interface powered by Google Gemma, accessible through the Ollama platform.

**Key Features:**

* **Contextual Question Answering:** Can answer questions based on provided context, quotes, or the content of a text file.
* **Stateless Conversations:** This is designed for single-turn questions and answers; it does not retain information from previous interactions, i.e., does not remember past chat content.
* **Ideal for Simple Q&A:** Best suited for straightforward questions and retrieving information.

**How to Use:**

* **Example Usage:** See the `ask_ai()` function in the Python code for an example of how to interact with the program.
* **Start a Session:** Running the script with the command `python3 simple_gemma_ai.py` will initiate a simple question and answer session. You can then ask questions like "List all male cats in my house."

**Underlying Technology:**

This program leverages the Google Gemma language model, which is run locally using the Ollama library.


## simple_gemma_ai_ja.py
A Japanese version of simple_gemma_ai.py. simple_gemma_ai.pyの日本語版。

例として、`python3 simple_gemma_ai_ja.py`で走らせて、質問してみてください。我が家の猫に関する情報をコンテクストとして与えているので、「我が家でオスの猫は？」などの質問をしてみてください。

