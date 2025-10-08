A simple ai assistant for home care medics created using python and the langchain framework which can answer questions based on the context of the website.
It uses llama3.1 via olllama for generating embeddings which enables it to run without API Keys for paid services

It has 4 processes:

1 It loads the content directly from HCM url using a webloader.

2 breaks the website's text into smaller chunks and converts them into numerical representation for embeddings using a text_splitter and stores in a vector store.

3 Retrieves relevant info when you ask a question by converting your question into an embedding and uses the vector store to find the most relevant chunks of text from the website.

4 It finally sends your question to llama3.1 llm which generates a natural language answer  based on the provided context.
