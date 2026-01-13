def main():

    print("Hello from arc!")

def test_ARCModule():
    import numpy as np
    from ARCDataClasses import ARCProblemSet
    from tensordict import TensorDict
    from jaxtyping import Int
    from arc.ARCModule import MultiTaskEncoder, positional_encodings
    from arc.ARCNetworks import Encoder, Decoder, AttributeHead
    from typing import Dict
    import torch
    import lightning as L


    GRID_SIZE: Int = 30*30
    ATTENTION_SIZES: Int = (32, 64, 128)
    MODEL_DIM: Int = 2048
    HIDDEN_ATTR_SIZE: Int = 96
    BATCH_SIZE: Int = 31
    EPOCHS: Int = 1
    ATTRIBUTES: Dict[str, Int] = { # TODO: There must be a better way to do this, perhaps dynamically from the dataset itself
        "area": 1,
        "grid_size": 2,
        "num_colors": 1,
        "color_map": 10
    }

    autoencoder = MultiTaskEncoder(
        Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM), 
        Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE),
        {
            key: AttributeHead(key, MODEL_DIM, HIDDEN_ATTR_SIZE, value) 
            for key, value in ATTRIBUTES.items()
        },
        learning_rate=1e-3,
        alpha=0.9,
        learning_rate_w=5e-3
    )

    training_dataset = ARCProblemSet.load_from_data_directory('training')

    def collate_fn(batch):
        names = [item["name"] for item in batch]
        padded_grids = torch.stack([item["padded_grid"] for item in batch], dim=0).reshape(-1, GRID_SIZE)
        return TensorDict(
            {
                "name": names,
                "embedding": None,
                "padded_grid": padded_grids,
                "encoded_grid": padded_grids + positional_encodings.reshape(-1, GRID_SIZE),
                "predicted_grid": None,
                "area": [item["meta"].area for item in batch],
                "predicted_area": None,
                "grid_size": [item["meta"].grid_size for item in batch],
                "predicted_grid_size": None,
                "num_colors": [item["meta"].num_colors for item in batch],
                "predicted_num_colors": None,
                "color_map": [item["meta"].color_map for item in batch],
                "predicted_color_map": None
            },
            batch_size=len(batch)
        )

    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    trainer = L.Trainer(max_epochs=EPOCHS)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # Take a single batch from the train_loader; print out the actuals and the predictions
    sample_batch = next(iter(train_loader))
    autoencoder.eval()
    with torch.no_grad():
        encoder_out = autoencoder.encoder(sample_batch)
        sample_batch["embedding"] = encoder_out["embedding"]
        decoder_out = autoencoder.decoder(sample_batch)
        reconstructed = decoder_out["predicted_grid"]

        attribute_predictions = {}
        for attr_key in ATTRIBUTES.keys():
            attr_head = getattr(autoencoder, f"attribute_head_{attr_key}")
            attr_out = attr_head(sample_batch)
            attribute_predictions[attr_key] = attr_out[f"predicted_{attr_key}"]
        
        sample_idx = 2
        sample_grid_size = sample_batch["grid_size"][sample_idx].squeeze().to(torch.int32)
        original = sample_batch["padded_grid"][sample_idx].reshape(30, 30)
        prediction = reconstructed[sample_idx].reshape(30, 30)
        
        print("=== GRID RECONSTRUCTION ===")
        print("Original grid:\n", original[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy())
        print("Predicted grid:\n", np.round(prediction[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy()))
        
        print("\n=== ATTRIBUTE PREDICTIONS ===")
        for attr_key in ATTRIBUTES.keys():
            predicted = attribute_predictions[attr_key][sample_idx].cpu().numpy()
            actual = sample_batch[attr_key][sample_idx]
            if hasattr(actual, 'cpu'):
                actual = actual.cpu().numpy()
            
            print(f"\n{attr_key.upper()}:")
            print(f"  Predicted: {predicted}")
            print(f"  Actual:    {actual}")


if __name__ == "__main__":
    test_ARCModule()