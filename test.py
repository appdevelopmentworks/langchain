import chainlit as cl
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory 
from langchain.memory import ConversationSummaryMemory 
from langchain.chains import ConversationChain 
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain 

chat = ChatOpenAI(
    model="gpt-4o",
    streaming=True
)

#ConversationSummaryMemoryを使用するように変更
memory = ConversationSummaryMemory(  
    llm=chat,
    return_messages=True,
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答をできるチャットボットです。メッセージを入力してください。").send()

@cl.on_message
async def on_message(message: cl.Message):
    #保存されているメッセージを取得する
    messages = chain.memory.load_memory_variables({})["history"]

    print(f"保存されているメッセージの数: {len(messages)}")
    #保存されているメッセージを1つずつ取り出す
    for saved_message in messages:
        #保存されているメッセージを表示する
        print(saved_message.content)

    result = chain(message.content)

    await cl.Message(content=result["response"]).send()