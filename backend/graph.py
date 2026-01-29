import os
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

# --- State Definition ---
class GraphState(TypedDict):
    question: str
    original_question: str
    generation: str
    documents: List[str]
    thoughts: List[str]
    retry_count: int

# --- Grader Models ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucination(BaseModel):
    """Binary score for hallucination check in generation."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# --- Nodes Implementation ---

def retrieve(state):
    """Retreive documents from Qdrant"""
    print("---RETRIEVING---")
    question = state["question"]
    thoughts = state.get("thoughts", [])
    thoughts.append("Searching course materials for relevant context...")
    
    client = QdrantClient(path="./qdrant_db")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()
    documents = retriever.invoke(question)
    # Ensure original_question is set from the start
    orig_q = state.get("original_question") or question
    return {"documents": documents, "thoughts": thoughts, "question": question, "original_question": orig_q}

def grade_documents(state):
    """Determines whether the retrieved documents are relevant."""
    print("---CHECKING RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    thoughts = state["thoughts"]
    thoughts.append("Grading retrieved segments for relevance...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a technical grader assessing relevance. 
    1. If the user asks for a specific Week (e.g., 'Week 5') and the document does NOT mention it, grade as 'no'.
    2. For conceptual questions (e.g., 'SQL to Semantic'), if the document discusses these technical topics, grade as 'yes'.
    3. The goal is to avoid hallucinating info for non-existent weeks, while allowing retrieval for valid technical concepts in the curriculum."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # Use original_question for strict relevance grading
    orig_q = state.get("original_question", question)
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": orig_q, "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
            
    if not filtered_docs:
        thoughts.append(f"No documents found mentioning specific constraints of: '{orig_q}'")
    else:
        thoughts.append(f"Found {len(filtered_docs)} relevant segments for '{orig_q}'.")
        
    return {"documents": filtered_docs, "thoughts": thoughts}

def generate(state):
    """Generate answer"""
    print("---GENERATING---")
    question = state["question"]
    documents = state["documents"]
    thoughts = state["thoughts"]
    thoughts.append("Synthesizing answer based strictly on course material...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            name = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            formatted.append(f"SOURCE: {name} (Page {page})\nCONTENT: {doc.page_content}")
        return "\n\n".join(formatted)

    prompt_template = """You are a Senior AI Research Scientist and Architect acting as a technical tutor for the SupportVector Course. 
    Your goal is to provide high-fidelity, academically rigorous explanations based EXCLUSIVELY on the provided context.

    CONTEXT: {context}
    
    USER QUESTION: {question}

    INSTRUCTIONS:
    1. **Strict Grounding**: Use ONLY the provided CONTEXT. If the information is not present, or if the user asks about a week not covered (e.g., 'Week 5'), explicitly state: 'I'm sorry, my current repository of course materials only covers Weeks 1 through 4. I do not have technical data for [Week X].'
    
    2. **Technical Depth**: Do not provide surface-level summaries. Dive deep into:
       - **Architectural Nuance**: Explain the underlying structures (e.g., vector spaces, transformer blocks, retrieval chains).
       - **Performance Trade-offs**: Discuss why one method might be preferred over another (e.g., latency vs. accuracy, cost of compute).
       - **Mathematical/Algorithmic Logic**: If the context mentions specific algorithms or logic (e.g., semantic chunking markers, embedding dimensions), explain them in detail.

    3. **Professional Terminology**: Use industry-standard technical vocabulary (e.g., "context window saturation," "high-dimensional manifold," "retrieval precision," "zero-shot reasoning").

    4. **Structure & Length**: 
       - Your response must be at least 15 technical lines long.
       - Use Markdown headers (###) to separate **Theoretical Foundation** from **Practical Implementation**.
       - Use bullet points for technical specifications.

    5. **Reference Policy**: 
       - DO NOT provide inline citations (e.g., no [Source, Page]). 
       - DO NOT mention PDF names. 
       - References are handled by a separate UI component.

    Technical Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    rag_chain = prompt | llm | StrOutputParser()
    
    if not documents:
        # Strictly use the original intent for the refusal
        user_intent = state.get("original_question", question)
        generation = f"I'm sorry, I could not find any specific information about '{user_intent}' in the course materials. Based on my database, the materials primarily cover LLM architecture, Semantic Search, and Vector Embeddings for Weeks 1 through 4. I do not have info for other weeks."
    else:
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
        
    retry_count = state.get("retry_count", 0) + 1
    return {"generation": generation, "thoughts": thoughts, "retry_count": retry_count}

def transform_query(state):
    """Transform the query to produce a better search."""
    print("---TRANSFORMING QUERY---")
    question = state["question"]
    thoughts = state["thoughts"]
    thoughts.append("Optimizing search query for better results...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    system = """You are a query assistant that rewrites a user question to better retrieve info from a vector store. \n
    CRITICAL: YOU MUST PRESERVE all specific constraints (Week numbers, specific terms like 'manifolds', 'transformers', etc.). \n
    Do not generalize the question too much. If the user asks for Week 5, the new query MUST include 'Week 5'."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Original question: {question}"),
    ])
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
    
    retry_count = state.get("retry_count", 0) + 1
    return {"question": better_question, "thoughts": thoughts, "retry_count": retry_count}

# --- Conditional Edges ---

def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a query."""
    if not state.get("documents"):
        if state.get("retry_count", 0) >= 2:
            return "generate"
        return "transform_query"
    return "generate"

def grade_generation_v_documents_and_question(state):
    """Determines whether the generation is grounded in the document and answers question."""
    print("---CHECKING HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    thoughts = state["thoughts"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Hallucination Grader
    hallucination_grader = llm.with_structured_output(GradeHallucination)
    hallucination_system = "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts."
    hall_prompt = ChatPromptTemplate.from_messages([
        ("system", hallucination_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hall_chain = hall_prompt | hallucination_grader
    
    # Answer Grader
    answer_grader = llm.with_structured_output(GradeAnswer)
    answer_system = """You are a grader assessing whether an answer accurately addresses a user question. \n
    CRITICAL: If the user asks for information about a specific thing (e.g. 'Week 5') and the answer provided does not actually confirm it is about that specific thing, or is just general information, you MUST grade it as 'no'."""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", answer_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    answer_chain = answer_prompt | answer_grader

    # Check original question for validity
    orig_q = state.get("original_question", question)
    score = hall_chain.invoke({"documents": documents, "generation": generation})
    
    if score.binary_score == "yes":
        thoughts.append("No hallucinations detected. Checking if specific question is answered...")
        score = answer_chain.invoke({"question": orig_q, "generation": generation})
        if score.binary_score == "yes":
            thoughts.append("Success! Final answer verified.")
            return "useful"
        else:
            thoughts.append(f"Answer failed to address specific constraint: '{orig_q}'")
            if state.get("retry_count", 0) > 2:
                thoughts.append("Max retries reached. Returning direct refusal.")
                return "useful"
            return "not useful"
    else:
        thoughts.append("Detected potential hallucination. Retrying generation...")
        if state.get("retry_count", 0) > 2:
            thoughts.append("Max retries reached despite potential hallucination. Returning best effort.")
            return "useful"
        return "not grounded"

# --- Graph Build ---x

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not grounded": "generate",
        "not useful": "transform_query",
        "useful": END,
    },
)

app = workflow.compile()
