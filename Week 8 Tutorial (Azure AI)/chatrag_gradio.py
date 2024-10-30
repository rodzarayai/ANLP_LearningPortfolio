import gradio as gr
import random
import time

from helpers.rag import get_augmented_generation, GROUNDED_PROMPT

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        # bot_message = random.choice(
        #     [f"How are you? {message}", 
        #      f"I love you {message}", 
        #      f"I'm very hungry{message}"])

        bot_message = get_augmented_generation(message, GROUNDED_PROMPT)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()