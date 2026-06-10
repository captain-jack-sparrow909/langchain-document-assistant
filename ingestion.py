import asyncio
import os
import ssl
import certifi

from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# CharacterTextSplitter splitter uses a single separator (or a list of separators in a straightforward way) to split text into chunks of a target size.
# RecursiveCharacterTextSplitter -> Instead of relying on a single separator, it tries a hierarchy of separators, recursively splitting until chunks fit within the desired size.
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)


# configure SSL context to use certifi certificates, this is needed because we're going to use 
# TavilyCrawl to scrape websites and it uses requests library which doesn't 
# use the system's CA bundle by default.

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE "] = certifi.where()
os.environ[" REQUESTS_CA_BUNDLE"] = certifi.where()


# embeddings:
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_max_seconds=10)

# vector store:
vector_store = PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings)

# tavily: below is to help an LLM search, explore, and extract structured content from the web.
tavily_extract = TavilyExtract()   # This tool is used for extracting clean, readable content from a specific URL.
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
# This is for mapping a website’s structure (like crawling links in a controlled way).
# Parameters:
# max_depth=5 → how deep it follows links (link → link → link…)
# max_breadth=20 → how many links per page it explores
# max_pages=1000 → hard limit on total pages visited

tavily_crawl = TavilyCrawl()
# This is the full crawling tool.
# What it does:
# Systematically visits multiple pages
# Extracts content from each page
# Can combine with extraction + mapping logic



# utitlity functions:
async def index_document_async(docs, batch_size=50):
    """Process documents in batches and add to vector store asynchronously."""
    log_header("INDEXING DOCUMENTS INTO VECTOR STORE")
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    # async function to add batches to the vector store with error handling
    async def add_batch_to_vector_store(batch, batch_number):
        """Add a batch of docs to the vector store."""
        try:
            await vector_store.aadd_documents(batch) # aadd_documents is the async version of add_documents, it allows us to add documents to the vector store without blocking the main thread, which is important for performance when dealing with large datasets.
            log_success(f"Vector Store: Successfully indexed batch {batch_number}. Documents indexed: {len(batch)}")
        except Exception as e:
            log_error(f"Vector Store: Error indexing batch {batch_number} - {str(e)}")
            return False
        return True

    # process all the batches concurrently :
    tasks = [add_batch_to_vector_store(batch, idx + 1) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successes and failures:
    success_count = sum(1 for result in results if result is True)
    failure_count = len(results) - success_count
    log_info(f"Indexing Summary: Successes: {success_count}, Failures: {failure_count}")



# main function:
async def main():
    """Main function to perform web crawling, content extraction, and vector store ingestion."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("TavilyCrawl: Starting to crawl the documentation from https://python.langchain.com/", Colors.PURPLE)

    # Crawl the website and extract content
    try:
        res = tavily_crawl.invoke({
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
            "instructions": "content on ai agents"  # needed to let tavily know what to look for when crawling and extracting, this is important to get relevant content and avoid noise
        })
        all_docs = [ Document(page_content=doc["raw_content"], metadata={"source": doc["url"]}) for doc in res["results"] ]
        log_success(f"TavilyCrawl: Successfully crawled and extracted content from the documentation. Total pages crawled: {len(all_docs)}")

        # splitting documents into chunks:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        splitted_docs = text_splitter.split_documents(all_docs)
        log_success(f"TextSplitter: Successfully split documents into chunks. Total chunks created: {len(splitted_docs)}")

        # index documents into vector store:
        await index_document_async(splitted_docs, batch_size=500)


    except Exception as e:
        log_error(f"TavilyCrawl: Error during crawling - {str(e)}")
        return


if __name__ == "__main__":
    asyncio.run(main())

