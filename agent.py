import asyncio
from typing import List, Tuple
import json
import requests
from config import settings

from embeddings import embed
from cache import vector_cache
from search import search_and_scrape
from summarizer import summarize


async def ask_agent(query: str, profile: dict) -> str:
    """Main orchestrator – fully async."""
    # Check if vector database is populated, if not populate it
    if vector_cache.index.ntotal == 0:
        print("Vector database is empty. Populating with BFSI data...")
        process_finco_data()
        print(f"Vector database now contains {vector_cache.index.ntotal} products")
    
    # Search vector database for relevant BFSI products
    query_embedding = embed(query)
    vector_results = vector_cache.search(query_embedding, k=100)
    
    print(f"Found {len(vector_results)} relevant products for query: '{query}'")
    
    if not vector_results:
        return "I couldn't find any relevant BFSI products in our database for your query.", []
    
    # Format the vector search results for LLM analysis
    vector_data = []
    for score, product_info in vector_results:
        vector_data.append(f"Product (Similarity: {score:.3f}): {product_info}")
    
    vector_summary = "\n".join(vector_data)
    
    # Combine profile, query, and vector database results for LLM analysis
    llm_input = {
        "profile": profile,
        "query": query,
        "vector_database_results": vector_summary
    }
    llm_response = await analyze_with_llm(llm_input)

    return llm_response

async def analyze_with_llm(input_data: dict) -> str:
    """Send data to LLM for analysis and response generation using Hugging Face Inference API."""
    # Format the prompt for BFSI product recommendations
    prompt = f"""You are a financial advisor helping a customer find the best BFSI (Banking, Financial Services, and Insurance) products.

Customer Profile: {input_data['profile']}
Customer Query: {input_data['query']}

Available BFSI Products from our database:
{input_data['vector_database_results']}

Based on the customer's profile and query, analyze the available products and provide a personalized recommendation. Consider:
1. How well each product matches the customer's needs
2. The similarity scores to understand relevance
3. Specific features that would benefit this customer

Please provide a clear, human-friendly response with:
- Your top recommendation(s) with reasoning
- Key benefits for this specific customer
- Any important considerations or alternatives

Response:"""

    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        # Hugging Face returns a list of generated texts
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        elif isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        else:
            return str(result)
    else:
        return f"Error from Hugging Face API: {response.status_code} {response.text}"

def process_finco_data():
    """Read finco.json and convert data to vector embeddings."""
    try:
        with open("data/finco.json", "r") as f:
            data = json.load(f)
        
        print(f"Loading {len(data)} BFSI products into vector database...")
        
        for i, item in enumerate(data):
            text_representation = f"{item['Category']} {item['Sub_Category']} {item['Provider']} {item['Product_Name']} {item['USP']} {item['Key_Features']}"
            embedding = embed(text_representation)
            vector_cache.add(embedding, text_representation)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} products...")
        
        print(f"Successfully loaded {len(data)} products into vector database")
        
    except FileNotFoundError:
        print("Error: data/finco.json file not found!")
        return
    except Exception as e:
        print(f"Error processing finco data: {e}")
        return

if __name__ == "__main__":
    process_finco_data()