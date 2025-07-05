import asyncio
from agent import ask_agent

async def test_agent():
    # Test query and profile
    query = "I need a credit card for travel"
    profile = {
        "age": 30,
        "income": "50000",
        "occupation": "Software Engineer",
        "travel_frequency": "frequent"
    }
    
    print("Testing agent with query:", query)
    print("Profile:", profile)
    print("-" * 50)
    
    try:
        response = await ask_agent(query, profile)
        print("Response:", response)
        print("URLs:", urls)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent()) 