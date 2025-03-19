# VeriSource AI - Retrieval-Augmented Research Assistant  

**VeriSource AI** is an **AI-powered academic research assistant** that leverages **Retrieval-Augmented Generation (RAG)** to provide **accurate, verifiable, and citation-backed knowledge**. It integrates **multiple scholarly sources**, including **Wikipedia, ArXiv, Google Scholar, PubMed, Open Textbook Library, and FAISS-based retrieval**, to deliver **credible academic research with direct source links**.

**Project Status: Currently in Development **  
- The system is functional but undergoing enhancements for **better ranking, filtering, and multi-source aggregation**.  
- Upcoming features include **user-friendly UI, advanced filtering, and improved citation handling**.  

---

## Key Features  
- **Retrieval-Augmented Generation (RAG)** for scholarly sources  
- **Multi-source integration** - Wikipedia, ArXiv, Google Scholar, PubMed, and more  
- **FAISS-powered document retrieval** for fast lookups  
- **Direct citation links** for verifiable knowledge  
- **AI-powered natural language responses** for research queries  
- **Seamless integration with Google Gemini AI** for summarization  

---

## Project Architecture  

1 **Data Retrieval:** Queries multiple sources via APIs  
   - **Wikipedia** (General Knowledge)  
   - **ArXiv** (Scientific Research)  
   - **Google Scholar** (Academic Papers)  
   - **PubMed** (Medical & Life Sciences)  
   - **Open Textbook Library** (Free Academic Books)  
   - **FAISS Vector Store** (Efficient Document Retrieval)  

2 **Processing & Structuring:** Extracts key content and formats citations  
3 **AI-Powered Summarization:** Uses **Google Gemini AI** for enhanced readability  
4 **Final Output:** Delivers structured answers with **clickable source links**  

---
