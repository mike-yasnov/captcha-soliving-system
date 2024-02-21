import torch 
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
model = model.to(device)
hidden_dim = model.config.hidden_size


transformation_chain = T.Compose(
    [
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

def extract_embeddings(batch: torch.tensor):
    """Utility to compute embeddings."""
    device = model.device

    new_batch = {"pixel_values": batch.to(device)}
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
    return embeddings

def compute_scores(emb_one: torch.tensor, emb_two: torch.tensor):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def fetch_similar(image: torch.tensor, embeds: torch.tensor, candidate_ids: list):
    """Fetches the similar images with `image` as the query."""

    new_batch = {"pixel_values": image.to(device)}

    with torch.no_grad():
        query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()

    sim_scores = compute_scores(embeds, query_embeddings)
    similarity_mapping = dict(zip(candidate_ids, sim_scores))
 
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    id = list(similarity_mapping_sorted.keys())[0]
    return id
