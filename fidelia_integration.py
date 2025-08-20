import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ApplicationBuilder
from openai import AzureOpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint="https://ai-bcds.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2025-01-01-preview"
)

# RAG System Functions
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed for input: {text[:30]}... | Error: {e}")
        return None

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def search_index_cosine(query, index, chunks, k=5):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Failed to get embedding for query.")
        return []
    query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
    query_vector /= np.linalg.norm(query_vector)

    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    similarities = distances[0]
    return list(zip(results, similarities))

def truncate_tokens(text, max_tokens=3000):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

def get_chatbot_answer(question, index, chunks, k=5):
    retrieved_context = search_index_cosine(question, index, chunks, k=k)
    context = "\n\n".join([chunk for chunk, score in retrieved_context])
    context = truncate_tokens(context, max_tokens=3000)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sales assistant supporting agents during client interactions. "
                                              "Your responses are based on Fidelidade's official documentation, the context provided and certified, accurate sources."
                                              "Your language is clear, polished, and client-friendly. "
                                              "If you don't know the answer to a question, respond with: "
                                              "'Thank you for this question. I will cross-check this internally and get back to you to ensure I provide accurate information.' "
                                              "Your scope is to provide information about Fidelidade's insurance products, their competitors products (comparison) and financial literacy, citing credible and nationally recognized sources from Portugal, including governmental publications. When answering questions about Financial literacy that are not in the documents, you will cite the source and only use credible portuguese, preferably governamental sources or from public institutions."
                                              "If a question is outside your scope, respond with: "
                                              "'This question is outside of my reach. I would love to help you with questions related to Fidelidade products, their competitors products (comparison) or financial literacy. Can I assist you within those boundaries?'"
                                              "Answer in the language you were prompted in: if in portuguese, answer with portuguese from Portugal, not Brazil"
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Chatbot failed to answer question: {question[:30]}... | Error: {e}")
        return ""

# Load the pre-built index and chunks
try:
    import pickle
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss_index.index")
    print("Successfully loaded chunks and index")
except Exception as e:
    print(f"Error loading chunks or index: {e}")
    exit(1)

async def handle_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message:
            # Get the user's question
            question = update.message.text
            
            # Print debug info
            print("\n=== New Message ===")
            print(f"From: {update.message.from_user.username}")
            print(f"Chat: {update.message.chat.title}")
            print(f"Question: {question}")
            
            # Get chatbot answer
            answer = get_chatbot_answer(question, index, chunks)
            
            # Send the answer back to the chat
            await update.message.reply_text(answer)
            
            print(f"Answer: {answer[:100]}...")
            print("==================\n")
            
    except Exception as e:
        error_msg = f"Error handling message: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        await update.message.reply_text("Sorry, I encountered an error processing your question.")

# --- Main bot setup ---
if __name__ == "__main__":
    try:
        # Option 1: Hardcode the token for local testing (not recommended for production)
        BOT_TOKEN = "7966302235:AAG5GTTNHOAA04Ep6durTzjf8ytAqqXJTQY"
        
        # Option 2: Load from environment variable (more secure)
        # BOT_TOKEN = os.getenv("BOT_TOKEN")
        if not BOT_TOKEN:
            try:
                with open("config.txt", "r") as f:
                    BOT_TOKEN = f.read().strip()
            except:
                print("⚠️ No BOT_TOKEN found in environment or config file!")
                exit(1)
                
        app = ApplicationBuilder().token(BOT_TOKEN).build()
        print(f"🔐 BOT_TOKEN: {BOT_TOKEN[:5]}...{BOT_TOKEN[-5:]}")

        # Add handler for ALL messages
        app.add_handler(MessageHandler(filters.ALL, handle_all_messages))

        print("🤖 Bot is running... Send a message to test.")
        print("Debug: Starting polling...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        print(f"Error starting bot: {str(e)}")
        exit(1)
        
