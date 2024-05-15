from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(
        model="gpt-4o",
        streaming=True
    )
    #
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはウォーレン・バフェット並みに優秀な投資家です、投資のアドバイスをしてください。",
            ),
            ("human", "{question}"),
        ]
    )
    #
    runnable = prompt | model | StrOutputParser()
    #
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()