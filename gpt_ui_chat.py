#
# This script sets up a simple AI assistant using Chainlit and GPT.
#
# It initializes a GPT model and defines event handlers for starting a chat 
# and handling incoming messages.
#
# - On chat start: the assistant sends a welcome message to the user.
# - On receiving a message: it checks for file attachments and notifies the user 
#   if any are found. It then forwards the user’s message to the GPT AI for 
#   processing, and finally displays the AI’s response in the Chainlit chat interface.
#
# The GptAI instance is initialized with `use_history=True`, allowing the assistant 
# to retain conversation context across turns.
#
# To run:
#     chainlit run gpt_ui_chat.py -h --host 0.0.0.0
#
# Then open your browser (if running on host 192.168.0.55, for example):
#     http://192.168.0.55:8000
#

import asyncio
import chainlit as cl
from gpt_ai import GptAi

# Initialize AI
ai = GptAi()

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

    # Asynchronous
    ''' This doesn't work
    placeholder = await cl.Message(content="").send() # Supposed to work....
    placeholder.content = await ai.ask_a(message.content)
    await placeholder.update()
    '''

    # Asynchronous
    # Create placeholder message
    placeholder = cl.Message(content="🤖 ...", author="GPT")
    await placeholder.send()

    # Show typing dots while waiting for response
    async def animate_dots():
        dots = 0
        while not done:
            placeholder.content = "🤖 " + "." * dots
            await placeholder.update()
            dots = (dots % 4) + 1
            await asyncio.sleep(0.5)

    done = False
    asyncio.create_task(animate_dots())

    # Get model's response
    response = await ai.ask_a(message.content)
    done = True

    # Show response
    placeholder.content = response
    await placeholder.update()


    '''
    # Synchronous
    placeholder = await cl.Message(content="....").send() # Show static typing dots

    # Ask AI
    response = ai.ask(message.content)

    # Send back to Chainlit frontend
    placeholder.content = response
    await placeholder.update()
    '''

