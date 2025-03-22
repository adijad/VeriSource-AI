import streamlit as st  # Import the Streamlit library
from RAG import wikipedia_with_clickable_link, retrieve_pubmed_articles  # Import functions from RAG.py

# Set the title of the app
st.title("RAG System - Document Retrieval Showcase")

# Provide a brief description
st.write("""
This app allows you to query Wikipedia and PubMed for relevant documents. 
Simply enter a query below, and we'll display the results along with clickable links to view more.
""")

# Add a text input for the query
query = st.text_input("Enter your query:")

# If the user enters a query, proceed to fetch results
if query:
    # Section for Wikipedia results
    st.subheader("Wikipedia Results")
    wikipedia_results = wikipedia_with_clickable_link(query)

    if wikipedia_results:
        for url in wikipedia_results:
            st.markdown(f"[Click here to view the Wikipedia article]({url})")
    else:
        st.write("No Wikipedia results found.")

    # Section for PubMed results
    st.subheader("PubMed Results")
    pubmed_results = retrieve_pubmed_articles(query)

    if pubmed_results:
        for idx, result in enumerate(pubmed_results):
            st.write(f"**Result {idx + 1}:**")
            st.write(result.page_content[:500] + '...')  # Show first 500 characters of the article

            # Safely extract the PubMed article ID and build the URL
            try:
                article_id = result.metadata.get('id', None)  # Try to fetch 'id' from metadata
                if article_id:
                    article_url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}"
                    st.markdown(f"[Click here to view the full article]({article_url})")
                else:
                    st.write("No valid PubMed ID found.")
            except Exception as e:
                st.write(f"Error: {str(e)}")
    else:
        st.write("No PubMed articles found.")
