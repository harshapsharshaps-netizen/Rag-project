import ollama
import numpy as np
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MMODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB=[]

def add_chunk_to_database(chunk):
    try:
        response = ollama.embed(model = EMBEDDING_MODEL,input= chunk)
        if 'embedding' in response:
            embedding = response['embedding']
        elif 'embedding' in response:
            embeddings = response['embeddings']
            if isinstance(embeddings[0],list):
                embedding = embeddings[0]
            else:
                embedding = embeddings
        else:
            print(f"unexpected response structure : {response.keys()}")
            raise ValueError("Could not find embedding in response")
        embedding_np = np.array(embeddings, dtype = np.float32)
        VECTOR_DB.append((chunk,embedding_np))
        print(f'Added chunk with embedding of size {embedding_np.shape}')
    except Exception as e:
        print(f'error adding chunk to database')
        

def cosine_similarity(a,b):
    a_np = np.array(a, dtype = np.float32)
    b_np = np.array(b, dtype = np.float32)

    dot_product = np.dot(a,b)

    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)

    if norm_a == 0 or norm_b == 0:
        return 0
    else:
        return dot_product/(norm_a * norm_b)
    

def retrieve(query , top_n=3):
    try:
        response = ollama.embed(model=EMBEDDING_MODEL , input=query)

        if 'embeddings' in response:
            embeddings = response['embeddings']
            if isinstance(embeddings[0], list):
                query_embedding = embeddings[0]
            else:
                query_embedding = embeddings
        else:
            print(f"unexpected response structure : {response.keys()}")
            raise ValueError("could not find embedding in response")
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        similarities = []
        for chunk,embdding_np in VECTOR_DB:
            similarity = cosine_similarity(query_embedding_np, embdding_np)
            similarities.append(chunk,similarity)

        similarity.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]
    except Exception as e: \
            print(f'Error retrieving:{e}')
        

def main():
    print("Loading dataset...")
    dataset = []
    
    with open('cat-facts.txt', 'r') as file:
        dataset =  file.readlines()
        
    dataset = [line.strip() for line in dataset if line.strip()]
    print (f'Loaded {len(dataset)} entries')
    
    print("Indexing dataset (this might take few minutes)...")
    for i, chunk in enumerate(dataset):
        add_chunk_to_database (chunk)
        print(f'Added chunk {1+1}/ {len(dataset)} to the database')
    while True:
        input_query = input('\n Ask me a question about cats (or type "exit" to quit):')
        if input_query.lower() == 'exit':
            break
        print('\n Retrieving Knowledge')
        retrieved_knowledge = retrieve (input_query)
        
        print('\n Retrieved Information')
        for chunk, similarity in retrieved_knowledge:
            print(f' (similarity: (similarity: 2)) (chunk)')
        context_text = "\n".join([f" - {chunk}" for chunk, similarity in retrieved_knowledge])
            
        instruction_prompt = f"""
        You are a helpful chatbot that knows about cats.
        Use only the following pieces of context to answer the question.
        Don't make up any new information: {context_text}"""

        print("\n Generating response....")
        stream = ollama.chat(
            model = LANGUAGE_MODEL,
            messages = [
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream= True
        )  
        print('\n Chatbot Response: ')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush = True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occured: {e}")



