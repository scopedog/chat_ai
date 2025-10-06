#
# This script sets up a simple AI assistant using Chainlit and Gemma.
#
# It initializes a Gemma model and defines event handlers for starting a chat 
# and handling incoming messages.
#
# - On chat start: the assistant sends a welcome message to the user.
# - On receiving a message: it checks for file attachments and notifies the user 
#   if any are found. It then forwards the user’s message to the Gemma AI for 
#   processing, and finally displays the AI’s response in the Chainlit chat interface.
#
# The GemmaAI instance is initialized with `use_history=True`, allowing the assistant 
# to retain conversation context across turns.
#
# To run:
#     chainlit run gemma_ui_chat.py -h --host 0.0.0.0
#
# Then open your browser (if running on host 192.168.0.55, for example):
#     http://192.168.0.55:8000
#

import chainlit as cl
from gemma_ai import GemmaAi
import ctx_dat

# Initialize GemmaAi
ai = GemmaAi()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="👋 Hi! I’m your AI assistant. Ask me anything!"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Check if any files are attached
    if message.elements:
        context_data = [element.path for element in message.elements]
        ai.add_context_data(context_data=context_data)

    # Show '...' (thinking dots)
    #bubble = await cl.Message(content="").send() # Supposed to work....
    bubble = await cl.Message(content="....").send()

    # Ask AI
    reply = ai.ask(message.content)

    # Send back to Chainlit frontend
    bubble.content = reply
    await bubble.update()
