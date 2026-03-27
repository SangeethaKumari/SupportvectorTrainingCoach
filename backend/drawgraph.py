import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    try:
        from backend.graph import app
    except ImportError:
        from graph import app

    png_data = app.get_graph(xray=True).draw_mermaid_png()
    
    output_path = "graph.png"
    with open(output_path, "wb") as f:
        f.write(png_data)
    
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())