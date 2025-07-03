import argparse, asyncio
from agent import ask_agent

parser = argparse.ArgumentParser(description="Web query agent CLI")
parser.add_argument("query", nargs="+", help="Search query")

if __name__ == "__main__":
    args = parser.parse_args()
    query = " ".join(args.query)
    result = asyncio.run(ask_agent(query))
    print(result)