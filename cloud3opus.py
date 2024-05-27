from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私はlaude-3-opusです、何でも質問してください。").send()
    model = ChatAnthropic(
        model="claude-3-opus-20240229",
        streaming=True
    )
    #
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは優秀なアシスタントです。",
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