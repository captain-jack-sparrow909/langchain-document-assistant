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
vector_store = PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embeddings=embeddings)

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
        all_docs = res["results"]
        log_success(f"TavilyCrawl: Successfully crawled and extracted content from the documentation. Total pages crawled: {len(all_docs)}", Colors.GREEN)
    except Exception as e:
        log_error(f"TavilyCrawl: Error during crawling - {str(e)}", Colors.RED)
        return


if __name__ == "__main__":
    asyncio.run(main())

