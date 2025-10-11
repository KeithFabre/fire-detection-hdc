import torch
import torchhd

# hyperparameters
DIMENSIONS = 10000  # 10,000-dimensional hypervectors
NUM_VECTORS = 1  # one hypervector per attribute

# creates random hypervectors for attributes
attributes = {
    "red": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS),
    "blue": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS),
    "pink": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS),
    "action_figure": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS),
    "hero": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS),
    "vehicle": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS),
    "princess": torchhd.random(num_vectors=NUM_VECTORS, dimensions=DIMENSIONS)
}

# encode toy by bundling attributes
def encode_toy(*attributes_to_combine):
    """
    Creates a toy representation by bundling attributes.
    torchhd.multiset performs hypervector bundling (summing).
    """
    # Sum the attributes to create a single hypervector
    return torch.sum(torch.stack(attributes_to_combine), dim=0)

# Create toy encodings
red_hero_action_figure = encode_toy(
    attributes["red"], 
    attributes["hero"], 
    attributes["action_figure"]
)

blue_hero_action_figure = encode_toy(
    attributes["blue"], 
    attributes["hero"], 
    attributes["action_figure"]
)

pink_princess = encode_toy(
    attributes["pink"], 
    attributes["princess"]
)

# stores the toys in a database (list)
toys_db = [
    red_hero_action_figure,
    blue_hero_action_figure,
    pink_princess
]
# Convert to tensor
toys_db_tensor = torch.stack(toys_db)

# queries the database using cosine similarity
def query_db(query_vector, db, threshold=0.5):
    """
    finds toys similar to the query vector.
    torchhd.cosine_similarity computes normalized dot product.
    returns indices of toys with similarity > threshold.
    """
    # Ensure query vector has same dimension as db
    if query_vector.dim() == 1:
        query_vector = query_vector.unsqueeze(0)
    similarities = torchhd.cosine_similarity(query_vector, db)
    return torch.where(similarities > threshold)[1].tolist()

# queries for princess-related toys
princess_query = attributes["princess"]
matches = query_db(princess_query, toys_db)

print(f"Matching toy indexes: {matches}")

# Additional functionality: Create and query a new toy
green_vehicle = encode_toy(
    torchhd.random_hv(NUM_HV, DIMENSIONS),  # green attribute
    attributes["vehicle"]
)

# adds to database
toys_db = torch.cat([toys_db, green_vehicle.unsqueeze(0)])

# queries for vehicle-related toys
vehicle_query = attributes["vehicle"]
matches = query_db(vehicle_query, toys_db)
print(f"Vehicle-related toys: {matches}")
