import gradio as gr
from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
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
from transformers import pipeline
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAConfig:
    """Configuration for the QA system"""
    pdf_urls: list
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
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        self.translator = GoogleTranslator()
        self.setup_qa_system()

    def setup_qa_system(self) -> None:
        """Set up the QA system with document loading and processing"""
        try:
            all_splits = []
            for pdf_url in self.config.pdf_urls:
                logger.info(f"Loading PDF document from {pdf_url}...")
                loader = OnlinePDFLoader(pdf_url)
                data = loader.load()

                logger.info("Splitting document into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                splits = text_splitter.split_documents(data)
                for split in splits:
                    split.metadata['source'] = pdf_url
                all_splits.extend(splits)

            logger.info("Creating vector store...")
            with SuppressStdout():
                vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=GPT4AllEmbeddings()
                )

            template = """You will work as a chatbot using RAG. You will be given a context and a question.
            If you don't know the answer based on the context, reply "I do not have the information related to the query",
            don't try to make up an answer, if the question is similar to the context suggest a few questions that are related to the context.
            Limit your answer up to 300 words. Be concise and to the point. Do not mention the context in your answer.
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
                return_source_documents=True,
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

            # Detectar el idioma
            detected_language = self.language_detector(query)[0]['label']
            logger.info(f"Detected language: {detected_language}")

            # Traducir usando Google Translate si no está en inglés
            if detected_language != 'en':
                translator = GoogleTranslator(source=detected_language, target='en')
                query = translator.translate(query)
                logger.info(f"Translated query to English: {query}")

            # Ejecutar la cadena QA
            result = self.qa_chain({"query": query})

            # Obtener la respuesta generada por Llama
            answer = result['result']

            # Limitar la respuesta a 400 tokens
            answer_tokens = answer.split()
            if len(answer_tokens) > 400:
                answer = ' '.join(answer_tokens[:400]) + '...'

            # Traducir la respuesta de vuelta al idioma original
            if detected_language != 'en':
                translator = GoogleTranslator(source='en', target=detected_language)
                answer = translator.translate(answer)

            # Extraer respuesta y metadatos de los documentos fuente
            source_documents = result.get("source_documents", [])
            sources = []
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                sources.append(source)

            # Formatear la respuesta
            sources_text = ", ".join(set(sources)) if sources else "Unknown"
            return f"{answer}\nSources: {sources_text}"
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"



def main():
    """Main function to run the Gradio interface"""
    config = QAConfig(
        pdf_urls=["https://d3n8a8pro7vhmx.cloudfront.net/foodday/pages/24/attachments/original/1341506994/FoodDay_Cookbook.pdf",
                  "https://arxiv.org/pdf/1706.03762"]
    )
    qa_system = DocumentQA(config)

    interface = gr.Interface(
        fn=qa_system.answer_query,
        inputs=gr.inputs.Textbox(label="Enter your query"),
        outputs=gr.outputs.Textbox(label="Answer"),
        title="Document QA with RAG",
        description="Ask questions based on the content of the loaded documents.",
        allow_flagging="never",
        theme=config.theme,
    )

    interface.launch()

if __name__ == "__main__":
    main()