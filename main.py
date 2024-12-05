import gradio as gr
from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAConfig:
    """Configuration for the QA system"""
    pdf_url: str
    chunk_size: int = 500
    chunk_overlap: int = 0
    model_name: str = "llama3.2"
    theme: str = "dark-peach"

class SuppressStdout:
    """Context manager to suppress stdout and stderr"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class DocumentQA:
    """Main class for document question-answering system"""
    
    def __init__(self, config: QAConfig):
        """Initialize the QA system with given configuration"""
        self.config = config
        self.qa_chain = None
        self.setup_qa_system()

    def setup_qa_system(self) -> None:
        """Set up the QA system with document loading and processing"""
        try:
            logger.info("Loading PDF document...")
            loader = OnlinePDFLoader(self.config.pdf_url)
            data = loader.load()

            logger.info("Splitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            all_splits = text_splitter.split_documents(data)

            logger.info("Creating vector store...")
            with SuppressStdout():
                vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=GPT4AllEmbeddings()
                )

            template = """You will work as a chatbot using RAG. You will be given a context and a question.
            If you don't know the answer based on the context, reply "I do not have the information related to the query",
            don't try to make up an answer, maybe suggest the user something related to the query based on the contexts.
            {context}
            Question: {question}
            Helpful Answer:"""
            
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            logger.info("Setting up LLM and QA chain...")
            llm = Ollama(
                model=self.config.model_name,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )
        except Exception as e:
            logger.error(f"Error setting up QA system: {str(e)}")
            raise

    def answer_query(self, query: str) -> str:
        """Process a query and return an answer"""
        try:
            if not query.strip():
                return "Please enter a valid query."
            logger.info(f"Processing query: {query}")
            result = self.qa_chain({"query": query})
            return result["result"]
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

def main():
    """Main function to run the Gradio interface"""
    config = QAConfig(
        pdf_url="https://d3n8a8pro7vhmx.cloudfront.net/foodday/pages/24/attachments/original/1341506994/FoodDay_Cookbook.pdf"
    )
    qa_system = DocumentQA(config)

    interface = gr.Interface(
        fn=qa_system.answer_query,
        inputs=gr.inputs.Textbox(label="Enter your query"),
        outputs=gr.outputs.Textbox(label="Answer"),
        title="Document QA with LangChain and Gradio",
        description="Ask questions based on the content of the loaded document.",
        allow_flagging="never",
        theme=config.theme,
    )

    interface.launch()

if __name__ == "__main__":
    main()