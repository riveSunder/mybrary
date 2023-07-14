#from transformers import pipeline, AutoTokenizer, AutoModel

def cls_pooling(model_output):

    cls_vectors = model_output.last_hidden_state[:,0,:]

    return cls_vectors

def mean_pooling(model_output):

    mean_vectors = model_output.last_hidden_state.mean(1)

    return mean_vectors

def l2_distance(query, embeddings): 
    # query is a 1xk vector, 
    # embeddings is n vectors in nxk matrix
    
    distances = ((query-embeddings)**2).sum(1, keepdims=True).sqrt().reshape(1,-1)

    return distances

def cosine_similarity(query, embeddings):
    
    q = query
    similarities = [(q @ e.t()) \
            / (q @ q.t() * e @ e.t()).sqrt() \
            for e in embeddings]

    return similarities

def cosine_distance(query, embeddings):

    similarities = cosine_similarity(query, embeddings)

    cosine_distances = [1.0 - (s/2. + 0.5) \
            for s in similarities]

    return cosine_distances
