import numpy as np

# creates a random high-dimensional vector
def create_random_vector(dimensions=10000):
  return np.random.choice([-1, 1], size=dimensions) # high dimension binary vector

# Define attributes as random vectors
attributes = {
    "red": create_random_vector(),
    "blue": create_random_vector(),
    "action_figure": create_random_vector(),
    "hero": create_random_vector(),
    "pink": create_random_vector(),
    "princess": create_random_vector(),
    "vehicle": create_random_vector(),
    "professor": create_random_vector(),
    "nce": create_random_vector(),
    "recreio": create_random_vector(),
    "caxambi": create_random_vector(),
    "student": create_random_vector(),
    "tijuca": create_random_vector()
}

# pra q serve esse * ?
def encode_toy(*attributes_to_combine):
    encoded_vector = np.sum(attributes_to_combine, axis=0)
    return encoded_vector / np.linalg.norm(encoded_vector)

red_hero_action_figure = encode_toy(attributes["red"], attributes["hero"], attributes["action_figure"])
blue_hero_action_figure = encode_toy(attributes["blue"], attributes["hero"], attributes["action_figure"])
pink_princess = encode_toy(attributes["pink"], attributes["princess"])
professor_claudio = encode_toy(attributes["professor"], attributes["nce"], attributes["caxambi"])
professor_gabriel = encode_toy(attributes["professor"], attributes["nce"], attributes["recreio"])
student_keith = encode_toy(attributes["student"], attributes["tijuca"], attributes["princess"])

# cosine similarity function
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# store toys in a db
toys_db = [red_hero_action_figure, blue_hero_action_figure, pink_princess, professor_claudio, professor_gabriel, student_keith]

# query the db
def query_db(query_vector, db, threshold=0.5):
    results = []
    for idx, toy in enumerate(db):
        similarity = cosine_similarity(query_vector, toy)
        if similarity > threshold:
            results.append(idx)
    return results

princess_query = attributes["professor"]
matches = query_db(princess_query, toys_db)

print(f"matching toy indexes: {matches}")
