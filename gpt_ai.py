# AI chat program using GPT such as gpt-4o-mini, gpt-5.
# Can answer a question using RAG and remember past chat content.
# Good for general Q&A or chatting.
# See example usage of GptAi at the bottom of this program after if __name__ == "__main__":
# You can use TXT, PDF, DOC, CSV, WEB (http), HTML, XLS, PPTX, ODT (LibreOffice) for RAG.
# For example:
#   context_data = ["aaa.pdf", "https://example.com", "bbb.doc"]
#   GptAi(context_data=context_data)
# This reads "aaa.pdf" and "bbb.doc" files and also accesses https://example.com, then uses their content with RAG.

import os
import sys
import asyncio
import time
import random
import string
import openai
from dotenv import load_dotenv
from loguru import logger
from typing import Optional
import ctx_dat

# Load OpenAI key, etc
if not load_dotenv():
    print('Error: ".env" is not found or not accessible') 
    sys.exit(1)

# Check OPENAI_API_KEY
if os.getenv("OPENAI_API_KEY") is None:
    print('Error: OPENAI_API_KEY is not set') 
    sys.exit(1)

from openai import OpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_chroma import Chroma
from chromadb.api.client import SharedSystemClient
import chromadb

# Global parameters
DEFAULT_MODEL = "gpt-4o-mini"
#DEFAULT_MODEL = "gpt-4.1-mini"
#DEFAULT_MODEL = "gpt-4.1-nano"

# My AI class
class GptAi:
    def __init__(
        self,
        docs: list[Document] = [], # Split documents
        combined_ctx: str = "",
        chunk_size: int = 1024,
        chunk_overlap: int = 120,
        pattern: Optional[str] = None,
        system_prompt: str = None,
        context_data: list[str] = None, # Sources of data (file paths, URLs)
                                    # If docs is not None, this is ignored
        max_docs: int = 0,
        use_history: bool = True, # Remember chat history
        max_history_len: int = 10, # Remember this # of past chats   
        session_id = None, # Session ID
    ):

        # Initialize data
        self.context_data = context_data
        self.use_history = use_history
        self.docs = None
        self.combined_ctx = ""
        data = []
        self.chat_history = []
        self.chat_history_len = max_history_len
        self.embeddings = None
        self.llm = None
        self.max_docs = max_docs
        self.db = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Store/generate session ID
        self.session_id = session_id if bool(session_id) else ''.join(random.choices(string.ascii_letters + string.digits, k=8))

        # Set system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a very kind assitant. Please answer user's questions."
                "\n\n"
                "{context}"
            )

        # Load context files, URLs
        if not bool(docs) and context_data is not None:
            docs_ctx = ctx_dat.load(
                            data=context_data,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap)
            docs = docs_ctx.docs
            combined_ctx = docs_ctx.combined_ctx
        self.docs = docs
        self.combined_ctx = combined_ctx

        # Initialize OpenAI
        self.openai = OpenAI()

        # Initialize embeddings
        # Currently, default model seems to be "text-embedding-ada-002"
        # Other options are: "text-embedding-3-large", "text-embedding-3-small"
        embeddings = OpenAIEmbeddings()
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embeddings = embeddings

        # Initialize db
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.db_collection_name = self.session_id
        #self.db_collection_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        #print("db_collection_name: " + self.db_collection_name)
        #self.db_client = chromadb.PersistentClient(path='chroma')
        self.db = Chroma(collection_name=self.db_collection_name,
                         embedding_function=embeddings,
                         #client=self.db_client)
                         )

        # Add docs, create context_content, etc 
        k = 4
        self.context_content = []
        docs_len = len(docs)
        logger.debug("docs_len: " + str(docs_len))
        if docs_len > 0:
            # Add docs to db
            try:
                self.db.add_documents(docs)
            except openai.RateLimitError as e:
                logger.error("self.db.add_documents: " + str(e))
                docs = None
                k = 1
            else:
                # Determine k for retriever
                if docs_len < 4:
                    k = docs_len

                # Save page_content as context_content for vectorizing
                for doc in docs:
                    self.context_content.append(doc.page_content)

        # Get retriever
        retriever = self.db.as_retriever(search_kwargs={"k": k})

        # Initialize llm
        llm = ChatOpenAI(model=DEFAULT_MODEL)
        self.llm = llm

        # Set LCEL question system prompt template
        condense_question_system_template = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        # Create LCEL question prompt
        condense_question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_question_system_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create Q and A prompt
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Build LCEL chains
        self.qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        if use_history:
            # Create history aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, condense_question_prompt
            )

            # Create chain
            self.chain = create_retrieval_chain(
                            history_aware_retriever, self.qa_chain)
        else:
            # Create non history aware retriever
            self.chain = create_retrieval_chain(retriever, self.qa_chain)

        logger.info("LLM has initialized")

    # Destructor
    def __del__(self):
        if logger is not None:
            logger.info("Class GptAi is destroyed")

        # Delete this data from Chroma
        #if self.db is not None:
            #print("Delete: " + self.db_collection_name)
            #self.db._client.delete_collection(self.db_collection_name)

    # Ask
    def ask(self, query: str):
        max_retries = 10
        num_retries = 0

        # Ask AI
        while num_retries < max_retries:
            try:
                # Ask AI
                chat_history = self.chat_history if self.use_history else []
                answer = self.chain.invoke(
                            {"input": query, "chat_history": chat_history}
                         )
                break
            except openai.RateLimitError as e:
                num_retries += 1
                time.sleep(20)

        # Too many retries
        if num_retries >= max_retries:
            raise RuntimeError("Too many invoke() retries")
        else:
            # Success
            res = answer['answer'].rstrip()

            # Adjust history
            if self.use_history:
                self.adjust_chat_history(query, res)

            return res

    # Ask (asynchronous)
    async def ask_a(self, query: str):
        max_retries = 10
        num_retries = 0

        # Ask AI
        while num_retries < max_retries:
            try:
                chat_history = self.chat_history if self.use_history else []
                answer = await self.chain.ainvoke(
                            {"input": query, "chat_history": chat_history}
                         )
                break
            except Exception as e:
                num_retries += 1
                time.sleep(20)

        # Too many retries
        if num_retries >= max_retries:
            raise RuntimeError("Too many invoke() retries")
        else:
            # Success
            res = answer['answer'].rstrip()

            # Adjust chat history
            if self.use_history:
                self.adjust_chat_history(query, res)

            return res

    # Adjust chat history
    def adjust_chat_history(self, query: str, res: str):
        if len(self.chat_history) >= self.chat_history_len:
            self.chat_history.pop(0)

        # Append answer to chat_history
        self.chat_history.extend([HumanMessage(content=query), AIMessage(content=res)])

    # Vectorize context data
    # Caution! If merge_all_content is True, this returns array of vector
    # If merge_all_content is False, it returns one vector (not array)
    def vectorize_context(self, merge_all_content=False):
        # Merge all content and create one page content
        if merge_all_content:
            content = ""
            for r in self.context_content:
                content += r
            # This returns one vector, not array
            #return self.embeddings.embed_query(content)
            return self.openai.embeddings.create(
                        input=[content],
                        model="text-embedding-3-small"
                        ).data[0].embedding
        else:
            # This returns array of vector
            #return self.embeddings.embed_documents(self.context_content)
            response = self.openai.embeddings.create(
                        input=self.context_content,
                        model="text-embedding-3-small"
                        )
            vectors = [d.embedding for d in response.data] 

    # Vectorize given single data
    def vectorize(self, data):
        return self.openai.embeddings.create(
                    input=[data],
                    model="text-embedding-3-small"
                    ).data[0].embedding

    # Summarize entire document
    # Since RAG cannot be used for summarization, we need to read entire context
    # or map'n reduce (summarize each chunk and summarize all of them)
    def summarize(self, prompt: str = None):
        # Init
        summary = None

        # Since it's very hard to tell if context fits llm limit,
        # we fitst try stuff chain
        # If failed, we try refine chain (very slow)
        try:
            summary = self.summarize_with_stuff(prompt)
        except Exception as e:
            logger.warning(e)

            # Then try refine chain - this takes long long time
            summary = self.summarize_with_refine(prompt)

        '''
        # Map & Reduce - doesn't work
        map_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("Summarize this chunk:\n\n{context}")
        ])

        combine_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("Combine these summaries:\n\n{context}")
        ])

        # 5. Use ChatOpenAI with gpt-4o-mini
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 6. Load summarization chain (map_reduce mode)
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            document_variable_name='context',
            verbose=True
        )

        # 7. Run the chain
        summary = chain.run(self.docs)
        '''

        '''
        map_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt),
            HumanMessagePromptTemplate.from_template("Please summarize the following text:\n\n{text}")
        ])

        # Reduce phase prompt (summarize the summaries)
        combine_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt),
            HumanMessagePromptTemplate.from_template("Please summarize the following summaries:\n\n{text}")
        ])

        # Load the chain with custom prompts
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt
        )

        #chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        #summary = chain.invoke(self.docs)

        # === Step 3: Run the chain ===
        summary = chain.invoke({"context": self.docs})
        #summary = chain.invoke(self.docs)
        '''

        return summary

    # Summarize entire document with stuff technique
    # Stuff chain - concatenates all docs and does not consider context size
    def summarize_with_stuff(self, prompt: str = None):
        # Check prompt
        if prompt is None:
            prompt = "Write a concise summary of the following:\\n\\n{context}"
        elif "{context}" not in prompt:
            # Prompt must contain "{context}"
            prompt += "\\n\\n{context}"

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages(
                [("system", prompt)]
            )

        # Instantiate chain and invoke
        res = None
        try:
            chain = create_stuff_documents_chain(self.llm, prompt_template)
            #res = chain.invoke({"context": self.docs})
            docs = [Document(page_content=self.combined_ctx, metadata={"source": "combined_ctx"})]
            res = chain.invoke({"context": docs})
        except Exception as e:
            raise Exception(e)

        return res

    # Summarize entire document with refine technique
    def summarize_with_refine(self, prompt: str = None):
        # Refine chain - takes long long time
        # Custom refine_prompt (used for each subsequent document)
        #refine_template = """
        #    Given the following existing answer: {existing_answer}
        #    And the following additional text: {text}
        #    Refine the answer, focusing on the most important information.
        #    """
        refine_template = """
            Given the following existing answer: {existing_answer}
            And the following additional text: {text}
            """

        refine_template += '\n' + prompt
        refine_prompt = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )

        # LLM model
        #llm = ChatOpenAI(model="gpt-4o-mini")

        # Load refine chain
        chain = load_summarize_chain(
            self.llm,
            chain_type="refine",
            refine_prompt=refine_prompt,
        )
            #question_prompt=question_prompt,

        # Run chain
        try:
            #res = chain.invoke({"input_documents": self.docs})
            docs = [Document(page_content=self.combined_ctx, metadata={"source": "combined_ctx"})]
            res = chain.invoke({"input_documents": docs})
        except Exception as e:
            raise Exception(e)

        return res['output_text']

    # Add context data
    def add_context_data(self, context_data: list[str]):
        docs_ctx = ctx_dat.load(
                        data=context_data,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap)
        docs = docs_ctx.docs
        self.db.add_documents(docs)
        #print(docs)
        self.docs.extend(docs)
        self.combined_ctx += docs_ctx.combined_ctx

        return docs_ctx

# Main
if __name__ == "__main__":
    # Get context data
    context_data = None
    if len(sys.argv) > 1:
        context_data = sys.argv[1:]

    # Set system prompt
    system_prompt = (
        "You are a very kind assitant. Please answer user's questions."
        "\n\n"
        "{context}"
    )

    # Initialize AI
    ai = GptAi(
        chunk_size=1024,
        chunk_overlap=128,
        context_data=context_data,
        system_prompt=system_prompt,
        use_history=True
    )

    # Ask question
    while True:
        question = ''
        res = None
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
                    print("\n* Asking AI....\n")
                    res = ai.ask(question)

                break

            # Append input to quesion
            question += q + '\n'

            # Check if question is just 'exit'
            if question == 'exit\n':
                terminate = True
                break

        if terminate:
            exit()

        # Print 'answer'
        if isinstance(res, dict):
            if 'answer' in res:
                print(res['answer'] + '\n')
            else:
                print("Error: Please specify key for 'res'")
        elif type(res) == str:
            print(res + '\n')
        else:
            print(f"Error: {type(res)} not supported")
