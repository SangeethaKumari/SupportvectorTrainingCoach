# SupportVector Training Coach: Architecture Overview 🛡️📚

The **SupportVector Training Coach** is an Agentic RAG (Retrieval-Augmented Generation) system specifically engineered to serve as a high-fidelity technical tutor for the "Large Language Models & AI Agents" course.

---

## 🏗 System Architecture Diagram

```mermaid
graph TD
    %% Frontend/Backend Boundary
    subgraph "Frontend (Next.js 15)"
        User([User Question]) --> UI[React UI Components]
        UI --> Stream[Streaming Token Listener]
        UI --> ThoughtPanel[Real-time Thought Trace]
        UI --> SourcePanel[Source Page Citations]
    end

    Stream -. "SSE (Server-Sent Events)" .-> FastAPI

    subgraph "Backend (FastAPI)"
        FastAPI[FastAPI Endpoint /chat] --> Graph[LangGraph Orchestrator]
        
        subgraph "Agentic Loop (LangGraph Workflow)"
            Graph --> Retrieve[Retrieve Chunks]
            Retrieve --> GradeDocs[Grade Relevance Grader]
            
            GradeDocs -- "Irrelevant" --> Transform[Transform Query]
            Transform --> Retrieve
            
            GradeDocs -- "Relevant" --> Generate[Generate Answer]
            
            Generate --> HallucinationAudit{Audit Logic}
            
            HallucinationAudit -- "Hallucinated" --> Generate
            HallucinationAudit -- "Not Useful" --> Transform
            HallucinationAudit -- "Grounded & Useful" --> FinalResponse[Final Streaming Output]
        end
    end

    %% Infrastructure
    Retrieve <--> Qdrant[(Qdrant Vector DB)]
    
    subgraph "AI Services (Google Gemini)"
        Retrieve -.-> Embed[Gemini-Embedding-001]
        GradeDocs -.-> Gemini[Gemini 1.5/2.0 Flash]
        Generate -.-> Gemini
        HallucinationAudit -.-> Gemini
        Transform -.-> Gemini
    end

    FinalResponse --> UI
    
    subgraph "Ingestion Pipeline (Offline)"
        PDFs[(Course PDFs)] --> Chunking[Semantic Chunking]
        Chunking --> Embed
        Embed --> Qdrant
    end
```

---

## 🛠 Technology Stack

### **Frontend**
*   **Framework**: Next.js 15 (App Router, TypeScript)
*   **Styling**: Tailwind CSS for high-fidelity technical UI.
*   **Animations**: Framer Motion for smooth state transitions and "thinking" indicators.
*   **Icons**: Lucide React for clean, minimalist iconography.

### **Backend**
*   **Framework**: FastAPI (Python)
*   **Package Manager**: `uv` (modern, ultra-fast Python dependency management).
*   **Streaming**: Server-Sent Events (SSE) to deliver real-time agent "thoughts" and generated tokens.

### **Intelligence & Orchestration**
*   **LLM**: **Google Gemini 3.0 Flash** (used for reasoning, grading, and generation).
*   **Orchestration**: **LangGraph** (StateGraph) for building robust, cyclic agentic workflows.
*   **Embeddings**: `models/gemini-embedding-001` for semantic representation.
*   **Vector DB**: **Qdrant** (Running in local path mode `./qdrant_db`).

---

## 🧠 The "Agentic Loop" Core Logic

Unlike standard RAG, the "Agentic" approach ensures high reliability and zero-hallucination through several self-correction nodes:

1.  **Semantic Retrieval**: Queries are vectorized and matched against high-dimensional manifold space in Qdrant.
2.  **Relevance Grader**: A binary classifier that filters out noise. If the retrieved context is irrelevant (e.g., asking about Week 10 when only Weeks 1–4 are loaded), the agent detects this.
3.  **Query Transformation**: If retrieval fails, the agent uses self-reflection to rewrite the user query (e.g., "What is Week 5?" becomes "SupportVector course curriculum week 5 syllabus") and retries.
4.  **Draft Generation**: Synthesizes a response using **strict grounding** instructions.
5.  **Multi-Stage Audit**:
    *   **Hallucination Check**: Compares the LLM response against the retrieved source facts.
    *   **Usefulness Check**: Verifies if the answer actually satisfies the specific user constraint.

## graph.py -Here is the step-by-step technical flow:

1. The Entry Point: 

retrieve
The graph begins by converting the user's natural language question into a vector and searching the Qdrant database. It retrieves the top technical chunks from your course PDFs.

Next Node: 

grade_documents
2. The Quality Control: 

grade_documents
This is where the "Agentic" part starts. A specialized Gemini 2.0 grader examines every retrieved chunk.

Logic: It looks for a semantic match. If you ask for "Week 5" and the chunk is about "Vectors in Week 1," the grader marks it as irrelevant.
Decision (Conditional Edge):
If relevant chunks exist: Proceed to 

generate
.
If no relevant chunks exist: Proceed to 

transform_query
.
3. The Recovery Step: 

transform_query
If the initial search failed, the agent doesn't give up. It asks Gemini to rewrite the user's query for better retrieval (e.g., "What is in Week 5?" might become "SupportVector course curriculum week 5 syllabus").

Next Node: It loops back to 

retrieve
 to try the search again with the new query.
4. The Production: 

generate
The agent takes the verified chunks and synthesizes a technical answer. It is strictly instructed to use only the provided context and to provide citations.

Next Node: 

grade_generation_v_documents_and_question
 (The Double-Check).
5. The "Zero-Hallucination" Gate
This is a complex conditional edge that performs two separate audits on the answer:

Hallucination Audit: It compares the answer to the source facts. If Gemini detects a fact not present in the sources, it sends the agent back to 

generate
 to "re-draft" the answer.
Usefulness Audit: It checks if the answer actually satisfies the user's question. If the answer is general but the question was specific, it sends the agent back to 

transform_query
 to find better data.
Exit: Only when the answer is both grounded and useful does the graph reach the END.
---

## 📂 Project Structure

```text
.
├── backend/
│   ├── main.py         # FastAPI Entry Point (Streaming logic)
│   ├── graph.py        # LangGraph Multi-Node Workflow
│   ├── ingestion.py    # PDF to Vector conversion
│   └── requirements.py # Dependency list
├── frontend/
│   ├── src/app/        # Next.js Pages & Layout
│   ├── src/components/ # Modular React Components
│   └── tailwind.config.ts
├── data/               # Course PDF source files
├── qdrant_db/          # Persistent local vector storage
└── .env                # API Keys and configuration
```
