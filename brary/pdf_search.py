import os
import argparse

from pdfminer.high_level import extract_text
from brary.vss import cls_pooling, mean_pooling, cosine_similarity, l2_distance  

import numpy as np

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
#text = extract_text("example.pdf")


def fragment_text(text: str, max_length: int, overlap: float=0.5):

    my_text = []

    step_size = int(max_length / (1.0 - overlap))

    for start in range(0, len(text)-max_length, step_size):
        my_text.append(" ".join(text[start:start+max_length]))

    return my_text

def get_embedding(text_list, model, tokenizer):
    pad_length = 128

    tokens = tokenizer(text_list, padding="max_length",\
            max_length=pad_length, truncation=True, \
            return_tensors="pt")

    token_items = {key:value for key, value in tokens.items()}
    
    output = model(token_items["input_ids"], \
            token_items["attention_mask"])
    embeddings = mean_pooling(output)

    return embeddings

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-q", "--query", type=str, nargs="+",\
        default=["Cogito ergot sum"],\
        help="query or queries to search for in pdf text"\
        "(separate by space and use single quote marks)")
    parser.add_argument("-d", "--directory", type=str,
        default="data")
    parser.add_argument("-i", "--input", type=str,\
        default="Evolution_of_Autopoiesis_and_Multicellularity_in_t.pdf",\
        help="filename for pdf of interest")
    parser.add_argument("-k", "--k", type=int,\
        default=3,\
        help="k for top-k matches to report")

    args = parser.parse_args()
    query = args.query
    filename = args.input
    path = args.directory
    k = args.k

    with torch.no_grad():
        path = "data"

        dir_list = os.listdir(path)

#        for filename in dir_list:
#            if filename.endswith("pdf"):

        vector_filepath = os.path.join(path, \
                f"{os.path.splitext(filename)[0]}.pt") 


        filepath = os.path.join(path, filename)

        text = extract_text(filepath)
        text_list = fragment_text(text.split(" "), 96, 0.5)

        my_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    
        tokenizer = AutoTokenizer.from_pretrained(my_model)
        model = AutoModel.from_pretrained(my_model)

        if os.path.exists(vector_filepath):
            print(f"loading pre-computed vectors from {vector_filepath}")
            embeddings = torch.load(vector_filepath)
        else:


            embeddings = None

            # single sample at a time (low ram on laptop)
            for ii in range(len(text_list)):

                if embeddings == None:
                    embeddings = get_embedding(text_list[ii:ii+1], model, tokenizer)
                else:
                    embedding = get_embedding(text_list[ii:ii+1], model, tokenizer)
                    embeddings = torch.cat([embeddings, embedding]) 
            

            torch.save(embeddings, vector_filepath)

        query_again = True
        while query_again:
            query_embeddings = get_embedding(query, model, tokenizer)

            cosine_matrix = torch.zeros(query_embeddings.shape[0], \
                    embeddings.shape[0])
            l2_matrix = torch.zeros(query_embeddings.shape[0], \
                    embeddings.shape[0])

            for index, query_embedding in enumerate(query_embeddings):
                cosine_similarities = torch.tensor(cosine_similarity(query_embedding, embeddings))
                l2_distances = l2_distance(query_embedding, embeddings)

                cosine_matrix[index,:] = cosine_similarities #torch.tensor(cosine_similarities)
                l2_matrix[index,:] = l2_distances

                cosine_indices = list(np.argsort(cosine_similarities))
                l2_indices = list(np.argsort(l2_distances))
                cosine_indices.reverse()

                for kk in range(k):
                    # top_k best results

                    idx = cosine_indices[kk]

                    cosine_match = f"{kk}th best match for query {query[index]}"\
                            f"\n\t with cosine similarity {cosine_similarities[idx]:.3f}"\
                            f"\n\n {text_list[idx]}"

                    print(cosine_match)

                for kk in range(k):
                    # top_k best results

                    idx = cosine_indices[kk]

                    l2_match = f"{kk}th best match for query {query[index]}"\
                            f"\n\t with l2 distance {l2_distances[:, idx].item():.3f}"\
                            f"\n\n {text_list[idx]}"

                    print(l2_match)
                    
            query = [input("enter another query (0 to end)")]

            if query[0] == "0":
                query_again = False
            else:
                query_again = True
