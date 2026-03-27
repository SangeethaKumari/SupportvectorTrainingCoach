# RAG Pipeline Walkthrough: SupportVector Training Coach

This document provides a step-by-step walkthrough of how the backend processes data and answers user queries, starting from the preliminary document chunking phase all the way to the final streamed response.

## 1. Data Ingestion & Chunking (`ingestion.py`)
Before the system can answer queries, the raw course materials need to be processed so that the AI can efficiently search and understand them.

* **Loading:** The `PyMuPDFLoader` reads course PDFs from the `data/` directory, extracting raw text and metadata (like page numbers and source filenames).
* **Chunking:** A `RecursiveCharacterTextSplitter` processes the loaded text. It splits the massive documents into bite-sized segments (chunks) consisting of **1000 characters**. An **overlap of 100 characters** is maintained between consecutive chunks to ensure that context or sentences split at boundaries aren't lost.
* **Embedding & Indexing:** Each chunk is converted into a high-dimensional vector using Google's `gemini-embedding-001` model. These vectors, along with the chunk's text and metadata, are then upserted into a **Qdrant** Vector Database, which allows for extremely fast similarity searches later.

## 2. The LangGraph State Workflow (`graph.py`)
When a user asks a question, the request enters a cyclical workflow built using **LangGraph**. The state passes around the user's `question`, `original_question`, retrieved `documents`, generated `thoughts`, and tracks the `retry_count`.

### A. Retrieval (`retrieve` node)
The user's question is embedded into a vector, and the system queries the Qdrant database to retrieve chunks (documents) that are semantically similar to the question.

### B. Relevance Grading (`grade_documents` node)
To prevent the model from using irrelevant information, an LLM specifically grades the chunks that were just retrieved.
* **Filter:** It checks if the retrieved documents actually cover the specific constraints of the user's question (e.g., if the user asked about "Week 5", it checks if the document mentions "Week 5", or if it discusses relevant technical topics for conceptual questions).
* **Outcome:** Non-relevant documents are tossed out. If no documents are left, the system realizes it can't answer the question directly with the current search.

### C. Decision Point
Based on the results of the document grading, the graph chooses the next step:
* If documents *are* relevant, move to **Generation**.
* If *no* relevant documents are found, move to **Transform Query** (unless the max retry limit is reached).

## 3. Answer Generation (`generate` node)
The system takes the remaining relevant documents and provides them as strictly grounded `CONTEXT` to the primary generation LLM (a `gemini-3-flash-preview` model acting as a Senior AI Research Scientist).

* It synthesizes a highly technical answer using **only** the allowed context chunks. It is explicitly instructed not to hallucinate outer knowledge.
* This is exactly where our `final_answer` tag comes into play, ensuring that we only track the tokens of this specific response stream.

## 4. Verification & Validation (Conditional Graders)
After generating an answer, the system does not immediately accept it. It runs two rigorous background checks:

* **Hallucination Grader:** Does another pass over the generated text and the retrieved context chunks to ensure the LLM didn't invent any facts that weren't present in the documents.
* **Answer Grader:** Checks if the generated text actually answers the specific constraints of the user's original question.
* **Outcome:** If both pass, the graph successfully ends (`END`). If there's a hallucination, it loops back to re-generate. If the answer is safe but generally unhelpful, it triggers a query transformation.

## 5. Self-Correction (`transform_query` node)
If the retrieval failed or the answer wasn't useful, the system asks an LLM to "rewrite" the user's question into a better search query. It preserves specific entities (like "transformers" or "Week 5") but adjusts the phrasing to hopefully pull better semantic matches from the vector database. It then loops all the way back to the **Retrieval** phase.

## 6. Real-time Streaming (`main.py`)
While all of this complex looping and logic happens in the background, the FastAPI server keeps the frontend constantly updated. Using `astream_events()`:
* It intercepts the intermediate `thoughts` appended to the state array and pushes them to the user (e.g., *"Searching course materials..."*).
* It intercepts the metadata when documents are successfully graded, sending the source filenames and page numbers up to the UI.
* Finally, as string tokens stream out of the tagged generation LLM, it routes them directly to the frontend interface, providing that rapid, typewriter-style display.




Libraries at a glance

FastAPI
Web framework — defines routes, handles HTTP, returns responses. The engine the whole server runs on.

StreamingResponse
Keeps the HTTP connection open and pushes chunks of data to the browser in real time instead of waiting for the full answer.

CORSMiddleware
Allows the Next.js frontend (different port) to talk to this backend. Without it, the browser blocks cross-origin requests.

When a browser loads a webpage, say your Next.js frontend running on http://localhost:3000, it creates a security boundary around that origin. An origin is the combination of protocol + domain + port — so localhost:3000 is one origin and localhost:8000 (your FastAPI backend) is a completely different one.
By default, the browser blocks JavaScript from making requests to a different origin. This is called the Same-Origin Policy — it's a built-in browser security rule to prevent malicious websites from silently reading your data from other sites.

Pydantic BaseModel
Validates incoming JSON. ChatRequest guarantees message is a string before it touches any logic.
graph_app
The compiled LangGraph workflow — the entire RAG pipeline (retrieve → grade → generate → check). Imported here and driven by astream_events.
uvicorn
ASGI server that actually runs the FastAPI app. Called at the bottom with host="0.0.0.0" port=8000.
dotenv
Loads GOOGLE_API_KEY, QDRANT_URL, COLLECTION_NAME etc. from a .env file into os.environ so the graph can read them.







## In details workflow

## The state (GraphState) 
is the single object that flows through every node. It holds the current question, the original unmodified question, retrieved documents, the generated answer, a running list of thoughts (a human-readable log of what the system is doing), and a retry_count to prevent infinite loops.
retrieve connects to Qdrant — either a cloud instance (if QDRANT_URL is set) or a local path. It converts the question into a vector embedding using gemini-embedding-001, does nearest-neighbour search in your PDF chunk collection, and returns the top matching document chunks. It also locks in original_question here so rewrites downstream don't corrupt the original user intent.

grade_documents is the first quality gate. For each retrieved chunk, it asks Gemini: "is this document actually relevant to the question?" The system prompt has a critical rule baked in — if the user asked for a specific Week (e.g. "Week 5") and the chunk doesn't mention it, grade no. This prevents the system from hallucinating content for weeks that don't exist in your PDFs.
decide_to_generate is a simple conditional: if no documents survived grading, check retry_count. If under the limit, rewrite the query and try again. If over the limit, generate anyway (returning a "not found" message). 

If documents exist, proceed to generate.
generate is the answer node. It formats the filtered chunks with their source filenames and page numbers, then passes them through a strict RAG prompt that instructs Gemini to answer only from the provided context — and to explicitly refuse if the week isn't covered.

grade_generation_v_documents_and_question is the second quality gate, running two checks in sequence. First, a hallucination check: is the answer grounded in the retrieved chunks? Second, an answer quality check using original_question: did the answer actually address what the user asked? This uses the original question (not the rewritten one) to catch cases where a rewritten query drifts from intent. If either check fails and retry_count is still under the limit, it loops back — either to generate again (if hallucinated) or to transform_query (if the answer was technically grounded but missed the point).

The retry_count guard is what prevents the graph from spinning forever — any path that would loop increments the counter, and at > 2 the system cuts its losses and returns whatever it has.


## In details hallucination and quality checker flow

## Gate 1 — hallucination check. 
The prompt passes in the full set of retrieved document chunks plus the generated answer and asks: "Is this answer grounded in these facts?" The LLM returns a GradeHallucination with binary_score: "yes" or "no". This is a factual fidelity test — it's asking whether the generation invented anything that wasn't in the source material. If the answer says "Week 3 covers backpropagation" but none of the retrieved chunks mention backpropagation, that's a "no" — the model made something up. The graph routes back to generate to try again.

## Gate 2 — answer quality check.
 This only runs if Gate 1 passes. It takes original_question (not the possibly-rewritten question) and the generation, and asks: "Does this answer actually address what the user asked?" This catches a subtler failure: the answer is factually grounded but still misses the point. For example, the user asked "explain Week 5" and the system retrieved Week 3 content — the generation may be perfectly grounded in those chunks, but it doesn't answer what was asked. Gate 2 catches that and returns "not useful", routing to transform_query to rewrite and try a different retrieval.

## Why two gates instead of one? 
Because grounded-but-wrong is a different failure from hallucinated. Sending a hallucinated answer back through query rewriting doesn't help — the retrieval was fine, the generation just lied. Sending a correctly grounded but off-topic answer back through generation doesn't help either — the retrieved docs were the wrong ones to begin with. The two gates route to different fixes.
The retry cutoff. Both gates have a retry_count > 2 escape hatch. When it fires, they return "useful" regardless — not because the answer is actually good, but to prevent an infinite loop. The generate node has a hardcoded fallback message for this case: it explicitly tells the user the materials only cover Weeks 1–4.


Libraries at a glance
what each import does
FastAPI
Web framework — defines routes, handles HTTP, returns responses. The engine the whole server runs on.
StreamingResponse
Keeps the HTTP connection open and pushes chunks of data to the browser in real time instead of waiting for the full answer.
CORSMiddleware
Allows the Next.js frontend (different port) to talk to this backend. Without it, the browser blocks cross-origin requests.
Pydantic BaseModel
Validates incoming JSON. ChatRequest guarantees message is a string before it touches any logic.
graph_app
The compiled LangGraph workflow — the entire RAG pipeline (retrieve → grade → generate → check). Imported here and driven by astream_events.
uvicorn
ASGI server that actually runs the FastAPI app. Called at the bottom with host="0.0.0.0" port=8000.
dotenv
Loads GOOGLE_API_KEY, QDRANT_URL, COLLECTION_NAME etc. from a .env file into os.environ so the graph can read them.