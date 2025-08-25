# Assistente Fidélia - AI-Powered Chatbot for Commercial Agents

An AI-powered chatbot designed to support Fidelidade's insurance agents by providing instant, accurate answers to complex product questions using Retrieval-Augmented Generation (RAG) technology.

## Project Overview

This solution was developed as part of the Business Cases with Data Science course in the MSc in Data Science and Advanced Analytics program at NOVA IMS, Lisbon. The course provides hands-on experience with real-world cases presented by established clients. This particular case was proposed by Fidelidade Insurance Company. <br>

The client’s business need was to offload routine explanatory tasks from agents, such as addressing clients with low financial literacy, enabling agents to focus on high-value sales and relationship-building activities. To meet this need, we developed a chatbot based on a Retrieval-Augmented Generation (RAG) architecture, grounded in official PDF documentation to ensure compliant and precise responses.


## Repository structure
- `pdf_chatbot.folder` - knowledge base with pdf documents provided by the client
- `BCWDS_chatbot_code.ipynb` - main notebook code with full RAG and Text Preprocessing Pipeline + testing and evaluation of algorithm
- `fid.png` and `download.png` - images used for the aesthetic of the streamlit interface
- `fidelia.py` - code developed to host streamlit chatbot, connected to main notebook code
- `chunks.pkl` ana `faiss_index.index` - chunks of text and semantic search systemextracted from main notebook to be used on chatbot for consistency
- `fidelia_integration` - code available to enable integration on telegram for business

 
## Technical Approach

| Phase              |	          Techniques                                                           |
|--------------------|-------------------------------------------------------------------------------- |
|Data Preparation    | Text and Image Extraction, Cleaning, Chunking. API Client Setup. RAG Framework  |
|Feature Engineering | Tokenization. Building a searchable index (FAISS) for real-time semantic search |
|Model Selection     | GPT-4-o-mini within a RAG pipeline                                              |
|Evaluation	         | Cosine similarity, Manual Qualitative Evaluation                                |


## Key hosted chatbot Features

- RAG architecture for accurate, document-grounded responses
- User-friendly Streamlit interface
- Voice input capability
- Multi-language support
- Integration capabilities (Telegram, Business API)

## Technology Stack

- **Programming Language**: Python
- **AI Framework**: Retrieval-Augmented Generation (RAG)
- **APIs**: Open AI API (GPT-4-o-mini)
- **UI Framework**: Streamlit
- **Data Processing**: Text preprocessing, tokenization
- **Vector Database**: FAISS structure for semantic search
  

## Key Challenges
- Limited ability to tune the closed-source model, since it cannot be retrained.
- Designing and implementing features to the interface that maximize usability and provide added value
- Restricting the chatbot to relevant information while still enabling it to answer questions outside the knowledge base
- Ensuring the chatbot provides helpful, accurate answers strictly based on provided documents while gracefully handling out-of-scope queries to avoid "hallucination" in a regulated industry.


## Instructions to Run
- Make sure all files are on the same directory
- Do not rename the folder with the knowledge base documents
- First run main notebook - it will generate a new folder on the directory called "outputs"
- To run `fidelia.py`:
  - uncomment the first lines of code and do the installs on the terminal, if needed
  - make sure the directory is correct before running, oterwise, change directory on terminal with "cd *copied path directory*"
  - run the command " streamlit run fidelia.py" on terminal to host the interface locally
 
## Results
- **83-95% Cosine Similarity** - How semantically identical the chatbot's responses are to the perfect, expert-written answers
- **1.5M€ net savings** - Considering a conservative projection of 15min saved per agent in the network

## Key Observations and Future Improvements
- The solution was also developed for integration via telegram, or other hosted platforms via Business API
- Introduce user satisfaction surveys after each response, storing results in dataset for future analysis
- Incorporate additional language quality metrics for more robust evaluation
- Document reference system could be introduced to prioritize recent information over older as knowledge base scales

## Authors
- André Lourenço
- Emir Kamiloglu
- Manuel Andrade
- Rute Teixeira
- Victor Silva

### More details on https://rutemteixeira.github.io/generic/p8-chatbot.html
