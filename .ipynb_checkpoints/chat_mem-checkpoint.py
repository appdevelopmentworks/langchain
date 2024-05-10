import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

#モデルを設定
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

#バッファーを初期化
memory = ConversationBufferMemory(
    return_messages=True,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.message(content="私は栄養指導ができるチャットボットです。メッセージを入力してください").send()

@cl.on_message
async def on_message(message: str):
    memory_message_result = memory.load_memory_variables({})
    messages = memory_message_result['history']
    messages.appnd(HumanMessage(content=message))
    #
    result = chat(
        messages
    )
    #
    memory.save_context(
        {
            "input": message,
        },
        {
            "output":result.content,
        }
    )
    #
    await cl.Message(content=result.content).send()