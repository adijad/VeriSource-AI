# --------------------------------------------------------------
###Imports
# --------------------------------------------------------------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
# from langchain_community.utilities import GoogleSearchAPIWrapper
# from langchain_community.tools import GoogleSearchRun
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import requests
from xml.etree import ElementTree
import webbrowser
from dotenv import load_dotenv
from IPython.display import Markdown, display
from rich import print
from rich.console import Console
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import requests
from xml.etree import ElementTree
import webbrowser
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RAG_code')))
# from RAG import tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import openai
from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re
import uvicorn

# --------------------------------------------------------------
###Load environment variables
# --------------------------------------------------------------
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# --------------------------------------------------------------
### Initialize Wikipedia API Wrapper for Document Retrieval
#---------------------------------------------------------------

# class WikipediaRetriever:
# #     def __init__(self, top_k_results=3, doc_content_chars_max=500):
# #         self.wrapper = WikipediaAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max)
# #
# #     def search(self, query):
# #         # Retrieve Wikipedia content
# #         result = self.wrapper.run(query)
# #
# #         # Instead of returning the full content, extract only the URLs
# #         # urls = [f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}" for _ in range(len(result))]
# #         #
# #         # return {"references": urls}
# #         url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
# #
# #         return {"references": [url]}
# #
# # # Modify the Wikipedia tool to call this retriever
# # def wikipedia_with_clickable_link(query):
# #     retriever = WikipediaRetriever()
# #     result = retriever.search(query)
# #
# #     # Only return the references (URLs)
# #     references = result["references"]
# #
# #     # # Log or print the references
# #     # for ref in references:
# #     #     print(f"Source: {ref}")
# #
# #     return references

class WikipediaRetriever:
    def __init__(self, top_k_results=3, doc_content_chars_max=500):
        self.wrapper = WikipediaAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max)

    def search(self, query):
        result = self.wrapper.run(query)
        # Returning structured data with title, content, and URL
        articles = []
        for res in result:
            article = {
                "title": res['title'],
                "content": res['content'],
                "url": f"https://en.wikipedia.org/wiki/{res['title'].replace(' ', '_')}"
            }
            articles.append(article)
        return articles

def wikipedia_with_clickable_link(query):
    retriever = WikipediaRetriever()
    results = retriever.search(query)
    return results


# ## Test Wikipedia Tool
# query = "Quantum Computing"
# response = wikipedia_with_clickable_link(query)
#
# print(response)


# --------------------------------------------------------------
### Initialize Arxiv API Wrapper for Document Retrieval
# ---------------------------------------------------------------

# console = Console()
#
#
# class ArxivRetriever:
#     def __init__(self, top_k_results=3, doc_content_chars_max=200):
#         self.wrapper = ArxivAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max)
#
#     def search(self, query):
#         """Fetches ArXiv papers and returns both text and URLs."""
#         arxiv = ArxivQueryRun(api_wrapper=self.wrapper)
#         results = arxiv.run(query)
#         return results
#
#
# def arxiv_with_clickable_link(query):
#     retriever = ArxivRetriever()
#     results = retriever.search(query)
#
#     # Check if results are strings (plain text) or structured objects
#     if isinstance(results, list) and isinstance(results[0], str):  # If results are just text
#         for idx, paper in enumerate(results):
#             console.print(f"[bold green]ArXiv Response {idx + 1}: {paper[:300]}...[/bold green]\n")
#         return results
#
#     # If results are structured objects
#     for paper in results:
#         if hasattr(paper, 'entry_id'):  # Check if the object has 'entry_id'
#             url = paper.entry_id
#             console.print(
#                 f"[bold green]ArXiv Response: Title: {paper.title}\nSummary: {paper.summary[:300]}...[/bold green]")
#             console.print(f"[bold blue][link={url}]Click here to see the full paper[/link][/bold blue]\n")
#         else:
#             console.print(f"[bold red]Error: The retrieved data is not in the expected format.[/bold red]\n")
#
#     return results
#
#
# ### Test Arxiv Tool
# query = "Quantum Computing"
# response = arxiv_with_clickable_link(query)
#
# # Print raw response (just in case you want to check the text)
# print(response)

class ArxivRetriever:
    def __init__(self, top_k_results=3):
        self.wrapper = ArxivAPIWrapper(top_k_results=top_k_results)

    def search(self, query):
        arxiv = ArxivQueryRun(api_wrapper=self.wrapper)
        results = arxiv.run(query)
        # Returning structured data with title, summary, and URL
        articles = []
        for res in results:
            article = {
                "title": res.title,
                "summary": res.summary,
                "url": res.entry_id
            }
            articles.append(article)
        return articles


def arxiv_with_clickable_link(query):
    retriever = ArxivRetriever()
    results = retriever.search(query)
    return results

# --------------------------------------------------------------
### Initialize Google Scholar API Wrapper for Document Retrieval
#---------------------------------------------------------------

# class GoogleScholarRetriever:
#     def __init__(self):
#         self.wrapper = GoogleSearchAPIWrapper(
#             google_api_key=google_api_key,
#             google_cse_id=google_cse_id
#         )
#         self.search = GoogleSearchRun(api_wrapper=self.wrapper)
#
#     def search(self, query):
#         """Fetches Google Scholar search results and returns text and URLs."""
#         results = self.search.run(query)
#         return results
#
#
# def google_scholar_with_clickable_link(query):
#     retriever = GoogleScholarRetriever()
#     results = retriever.search(query)
#
#     for entry in results:
#         # Extract URL
#         url = entry['link']
#
#         # Display result in PyCharm terminal with Rich
#         console.print(
#             f"[bold green]Google Scholar Response: Title: {entry['title']}\nSnippet: {entry['snippet']}[/bold green]")
#         console.print(f"[bold blue][link={url}]Click here to see the article[/link][/bold blue]\n")
#
#     return results
#
# ### Test Google Scholar Tool
# query = "Quantum Computing"
# response = google_scholar_with_clickable_link(query)
#
# # Print raw response (just in case you want to check the text)
# print(response)

class GoogleScholarRetriever:
    def __init__(self):
        self.wrapper = GoogleSearchAPIWrapper(google_api_key=os.getenv("GOOGLE_API_KEY"),
                                              google_cse_id=os.getenv("GOOGLE_CSE_ID"))
        self.search = GoogleSearchRun(api_wrapper=self.wrapper)

    def search(self, query):
        results = self.search.run(query)
        # Returning structured data with title, snippet, and URL
        articles = []
        for entry in results:
            article = {
                "title": entry['title'],
                "snippet": entry['snippet'],
                "link": entry['link']
            }
            articles.append(article)
        return articles

def google_scholar_with_clickable_link(query):
    retriever = GoogleScholarRetriever()
    results = retriever.search(query)
    return results


# --------------------------------------------------------------
### Initialize PubMed
#---------------------------------------------------------------

######### Class: PubMedLoader (Fetching Articles)
class PubMedLoader:
    def __init__(self, query, top_k=3):
        self.query = query
        self.top_k = top_k
        self.api_key = os.getenv("PUBMED_API_KEY")
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    def fetch_article_ids(self):
        """Fetches PubMed article IDs for a given query."""
        params = {
            "db": "pubmed",
            "term": self.query,
            "retmode": "json",
            "retmax": self.top_k,
            "api_key": self.api_key
        }
        response = requests.get(self.base_url, params=params).json()
        return response.get("esearchresult", {}).get("idlist", [])

    def fetch_article_abstracts(self, article_ids):
        """Fetches full abstracts for given PubMed article IDs."""
        articles = []
        for article_id in article_ids:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={article_id}&retmode=text&rettype=abstract&api_key={self.api_key}"
            article_text = requests.get(url).text
            articles.append({"id": article_id, "content": article_text})
        return articles

    def load(self):
        """Fetches articles and returns them as raw text."""
        article_ids = self.fetch_article_ids()
        if not article_ids:
            return []

        return self.fetch_article_abstracts(article_ids)


###### Embedding Model & FAISS Initialization

# Initialize Google AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize FAISS as None (will be created when data is available)
vectordb = None
retriever = None

# Text Splitter for chunking abstracts before storing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


########## Function: Retrieve PubMed Articles

def retrieve_pubmed_articles(query):
    global vectordb, retriever

    # Check FAISS first
    if retriever:
        similar_docs = retriever.get_relevant_documents(query)
        if similar_docs:
            print("Retrieving from FAISS (Cached Results)")
            return similar_docs

    print("No Cached Results, Calling PubMed API...")

    # Fetch fresh articles from PubMed API
    pubmed_loader = PubMedLoader(query, top_k=3)
    docs = pubmed_loader.load()

    if not docs:
        print("No articles retrieved from PubMed.")
        return []

    print(f"Retrieved {len(docs)} articles from PubMed")

    # Displaying results with clickable links using Rich
    for doc in docs:
        article_id = doc['id']
        article_content = doc['content']
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}"

        # Render link with Rich
        # console.print(f"[bold green]PubMed Response: {article_content[:300]}...[/bold green]")
        # console.print(f"[bold blue][link={url}]Click here to see the full article[/link][/bold blue]\n")

    # Store new results in FAISS only if there are documents
    doc_objects = [Document(page_content=doc['content']) for doc in docs]

    if doc_objects:
        chunked_documents = text_splitter.split_documents(doc_objects)

        # Store in FAISS
        vectordb = FAISS.from_documents(chunked_documents, embeddings)
        retriever = vectordb.as_retriever()
        print("New results stored in FAISS for future queries.")

    return doc_objects

### Test PubMed Tool

# Test PubMed Retrieval
# query = "Artificial Intelligence in Healthcare"
# results = retrieve_pubmed_articles(query)
#
# # Print the response
# for idx, result in enumerate(results):
#     print(f"\n🔹 Result {idx+1}:\n{result.page_content[:500]}...")
#




# Wikipedia Tool
wikipedia_tool = Tool(
    name="Wikipedia_Search",  # ✅ Name must be valid for Gemini API
    func=wikipedia_with_clickable_link,  # Calls the function we modified
    description="Search for Wikipedia articles on a given topic. Returns both content and source URL."
)

arvix_tool = Tool(
    name="ArXiv_Search",
    func=arxiv_with_clickable_link,
    description="Search for academic research papers from ArXiv based on a given query. Use this tool for scientific topics."
)

google_scholar_tool = Tool(
    name="Google_Scholar_Search",
    func=google_scholar_with_clickable_link,
    description="Search for academic research papers from Google Scholar based on a given query. Use this tool for scientific topics."
)
# PubMed Tool
pubmed_tool = Tool(
    name="PubMed_Search",
    func=retrieve_pubmed_articles,  # Using the PubMed retrieval function
    description="Search for academic research papers from PubMed based on a given query. Use this tool for medical and scientific topics."
)


tools = [wikipedia_tool, pubmed_tool]




# ------------------------------
# Step 1: Initialize LLM and Prompt
# ------------------------------

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0, max_tokens=700)
updated_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant. When answering a question, always include both the content and the source URL if available."
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ------------------------------
# Step 2: Define Tools and Agent
# ------------------------------
agent = create_openai_tools_agent(llm, tools, updated_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

























