import os
import sys
from dotenv import load_dotenv

# Add current directory to path so we can import backend
sys.path.append(os.getcwd())

from backend.graph import app as agent

def test_query(question: str):
    load_dotenv()
    
    print(f"\n🚀 Sending Question: {question}")
    print("-" * 50)
    
    # Initial state
    inputs = {"question": question, "retry_count": 0, "thoughts": []}
    
    # Stream the events to see the "Agentic Loop" in action
    for output in agent.stream(inputs):
        for key, value in output.items():
            print(f"\n🧠 Node: {key}")
            if "thoughts" in value and value["thoughts"]:
                print(f"💭 Current Thought: {value['thoughts'][-1]}")
            if "generation" in value:
                print(f"📝 Draft Answer: {value['generation'][:100]}...")

    # Final Result
    final_state = agent.invoke(inputs)
    print("\n" + "="*50)
    print("✅ FINAL VERIFIED ANSWER:")
    print("="*50)
    print(final_state["generation"])
    print("\n📚 SOURCES USED:")
    for doc in final_state.get("documents", []):
        print(f"- {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")

if __name__ == "__main__":
    query = input("\n🔍 Enter a technical question about the LLM course: ")
    test_query(query)
