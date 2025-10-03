Project Requirements Summary

1. Core Objective The main goal is to build an automated system, or agent, to process a list of questions from a governance document. This system must answer the questions based on a provided set of context documents.

2. Inputs & Outputs

Input - Questions: The list of questions originates from a 4-page DOCX, which has been converted into a structured Markdown file containing 17 sections.

Input - Context: A wide variety of documents will be provided as the knowledge base for answering the questions.

Output 1 - Unanswerable Report: The system must generate a report listing all questions that cannot be answered from the provided context.

Output 2 - Answered Template: The system must create and populate a document or template with the questions that could be answered, along with their generated answers.

3. Functional Requirements

Answerability Check: A core function is to determine for each question whether it is answerable only from the given context documents.

Sequential Processing: The system should be designed to process the questions sequentially, one at a time.

Handling Diverse Questions: The solution must be capable of answering a mix of question types, including simple factual queries and more complex explanatory ones.

4. Architectural & Design Considerations

Agentic Approach: You are leaning towards an "agentic" design that can employ multiple search patterns or reasoning steps, rather than a simple "query-once-and-done" RAG system.

Hybrid Retrieval: You want to explore a hybrid retrieval strategy that combines semantic (vector) search with traditional keyword search to improve accuracy.

Vector Database: You are considering using Qdrant, specifically in its in-memory configuration for the initial build.

5. Technical Stack & Framework Evaluation

POC Constraints: For the initial Proof of Concept, the setup must be lightweight. It should rely on pip-installable libraries and in-memory processes, explicitly avoiding heavy dependencies like a dedicated Postgres server.

Frameworks: You are evaluating the best available frameworks for the job. Your search started with LlamaIndex and LangChain but has expanded to include a thorough evaluation of all top-tier options to find the best fit. You have also inquired about the roles of other tools like CrewAI and Instructor (formerly Pydantic AI).

Models & Services: Your initial thinking for the model stack involves using a managed service like AWS Bedrock, with Cohere for embeddings and a high-performance LLM like a Claude Sonnet model for generation, although this is not finalized.

WHat do you think, ponder on it for a bit.
