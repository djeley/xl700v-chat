from operator import itemgetter
from pathlib import Path
from typing import List

import chainlit as cl
from langchain.indexes import SQLRecordManager, index
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, DirectoryLoader
)
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

chunk_size = 1024
chunk_overlap = 50

embeddings_model = OpenAIEmbeddings()

DOC_STORAGE_PATH = "data"


def process_docs(doc_storage_path: str):
    path = (Path(__file__).parent / DOC_STORAGE_PATH).resolve()

    # Load PDF files.
    pdf_loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    pdf_documents = pdf_loader.load()

    # Configure text splitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split all docs.
    docs = text_splitter.split_documents(pdf_documents)  # type: List[Document]

    # Create Chroma vector store.
    vector_store = Chroma.from_documents(docs, embeddings_model)

    # Doc storage DB.
    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    # Index data from the loader into the vector store.
    index_result = index(
        docs,
        record_manager,
        vector_store,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return vector_store


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


doc_search = process_docs(DOC_STORAGE_PATH)
model = ChatOpenAI(model_name="gpt-4o", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    system = """
    You are a friendly assistant, guiding Honda XL700V and XL700VA motorcycle owners through the verbose user guide with short and simple answers.
    If you are asked to provide more detailed answers, you can do so.
    Always provide torque values when applicable.
    Reference section and page numbers when applicable.
    Don't make things up. If you don't know the answer to a question, say so.
    """

    human = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", human),
        ]
    )

    # RAG to populate the context from the question.
    context = (
            itemgetter("question")
            | doc_search.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'fetch_k': 50, 'lambda_mult': 0.1})
            | format_docs
    )

    # Main chain.
    chain = (
            RunnablePassthrough.assign(context=context)
            | prompt
            | model
            | StrOutputParser()
    )

    # Chat history using question/response (not context).
    ephemeral_chat_history_for_chain = ChatMessageHistory()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: ephemeral_chat_history_for_chain,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    cl.user_session.set("runnable", chain_with_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    async with cl.Step(type="run", name="XL700V/VA-User-Guide"):
        async for chunk in runnable.astream(
                {"question": message.content},
                config=RunnableConfig(configurable={"session_id": "unused"},
                                      callbacks=[
                                          cl.LangchainCallbackHandler()
                                      ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()
