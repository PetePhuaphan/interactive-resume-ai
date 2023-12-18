# Interactive Resume AI

## Project Overview
Interactive Resume AI is an innovative project designed to revolutionize the way resumes and professional profiles are accessed and interacted with. This project leverages the power of AI to create a dynamic and responsive experience for users seeking to explore professional profiles and resumes.

## Concept
At its core, the Interactive Resume AI is built to integrate a user's resume and profile information within a vector database (Pinecone, for this project) facilitating efficient and relevant data retrieval. The integration with OpenAI allows for an advanced understanding and processing of user queries, ensuring accurate and relevant responses. Furthermore, the project is brought to life through a Streamlit-based web interface, offering an interactive and user-friendly platform.

## How it Works
When a user interacts with the Chat AI on our website, their queries are processed to search for relevant terms within the Pinecone vector database. This database contains structured information from resumes and profiles. The script then constructs a prompt based on this information, which is sent to OpenAI for processing. The response from OpenAI is then displayed to the user, providing a seamless and intuitive interaction experience.

## Technology Stack
* **LangChain Library** -Facilitates the integration and interaction between different AI models and components. Key Features: Streamlines the development of language applications, supports chaining different AI models, provides tools for conversation handling and context management.

* **Pinecone Vector Database** - Stores and manages resume and profile data efficiently. Key Features: Fast search capabilities, scalability, and advanced vector indexing.

* **OpenAI Integration** - Processes natural language queries and constructs meaningful responses. Key Features: Advanced language understanding, context-aware processing.

* **Streamlit Web Interface** - Provides a user-friendly and interactive platform for users to interact with the AI. Key Features: Easy to build and deploy, supports rapid prototyping, highly interactive.

## Architecture
**Indexing**: a pipeline for ingesting data from a source and indexing it. This usually happen offline.

![rag_indexing](https://github.com/PetePhuaphan/interactive-resume-ai/blob/main/img/rag_indexing.png)

**Retrieval and generation**: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

![rag_retrieval_generation](https://github.com/PetePhuaphan/interactive-resume-ai/blob/main/img/rag_retrieval_generation.png)

source : [Langchain.com](https://python.langchain.com/docs/use_cases/question_answering/)

## Data Preparation with OpenAI Embeddings
The initial step in preparing the Interactive Resume AI involves splitting the documents into manageable chunks and creating embeddings for all the documents - resumes and profiles or any relevant documents - that are to be stored in the vector database. This is achieved using OpenAI's Embeddings API.

When we pass text to the OpenAI Embeddings API, it returns a series of vectors, each having 1536 dimensions. These vectors are arrays consisting of 1536 floating point numbers. They represent the semantic essence of the text in a high-dimensional space, capturing intricate details and nuances of the content in each document.

## Storing Data in Pinecone Vector Database
Once the embeddings are generated, they are inserted into the Pinecone vector database. This database is specifically configured with the same dimensionality of 1536 to match the embeddings created by OpenAI's API. The use of cosine similarity as the metric in Pinecone ensures that the similarity search is aligned with the nature of the embeddings, allowing for the most relevant and accurate matches based on the query.

* Vector Database - A vector database stores these vectors and allows for efficient querying and retrieval. It's optimized for the kinds of operations needed for high-dimensional vector data, which are not typically efficient in traditional relational databases.

* Similarity Search - One of the primary operations performed on a vector database is similarity search. This refers to finding the vectors in the database that are most similar to a given query vector.

## Accessing the Application
Open your web browser and navigate to [Interactive Resume AI](https://interactive-resume-ai.streamlit.app/). This link will take you directly to the application hosted on Streamlit.

When a query is input into the Chat AI on our website, it undergoes a process to find relevant terms within the Pinecone vector database. The database, equipped with these pre-processed embeddings, facilitates fast and precise information retrieval.

The script then constructs a prompt integrating the insights gathered from the database together with original query from user. This prompt is sent to OpenAI for processing. The response generated by OpenAI, which is based on this prompt, is then presented to the user. This entire workflow ensures that the user experience is seamless and intuitive, providing relevant and contextually accurate information in real-time.
