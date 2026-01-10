# Imports
import os
import glob
import torch
from tensordict import TensorDict
from lightning.pytorch import Trainer
from OutputGrid import MultiTaskEncoder, Encoder, Decoder, positional_encodings, AttributeHead
from ARCDataClasses import ARCProblemSet
from jaxtyping import Int

GRID_SIZE: Int = 30*30
ATTENTION_SIZES: tuple[Int, Int, Int] = (128, 256, 128)
MODEL_DIM: Int = 64
HIDDEN_ATTR_SIZE: Int = 128
OUTPUT_ATTR_SIZE: Int = 16
BATCH_SIZE: Int = 41
EPOCHS: Int = 10
TENSORDICT_PATH: str = "data/tensordict_training.pt"
CHECKPOINT_DIR: str = "lightning_logs/"

def get_latest_checkpoint(dir: str = CHECKPOINT_DIR) -> str | None:
	"""
    Retrieve the latest checkpoint file from the given directory
    """
	checkpoint_paths = glob.glob(os.path.join(dir, "version_*/checkpoints/*.ckpt"))
	if not checkpoint_paths:
		return None
	return max(checkpoint_paths, key=os.path.getctime)

def build_tensordict():
	training_dataset = ARCProblemSet.load_from_data_directory('training')
	def collate_fn(batch):
		names = [item["name"] for item in batch]
		padded_grids = torch.stack([item["padded_grid"] for item in batch], dim=0).reshape(-1, GRID_SIZE)
		return TensorDict(
			{
				"name": names,
				"padded_grid": padded_grids,
				"encoded_grid": padded_grids + positional_encodings.reshape(-1, GRID_SIZE),
				"embedding": None,
				"reconstructed_grid": None,
                "attributes": torch.stack([item["attributes"] for item in batch], dim=0).reshape(len(batch), -1).to(torch.float32),
                "predicted_attributes": None
			},
			batch_size=len(batch)
		)
	# Save the tensordict for future use
	tensordict = collate_fn(training_dataset)
	torch.save(tensordict, TENSORDICT_PATH)
	return tensordict

def main():
	if os.path.exists(TENSORDICT_PATH):
		tensordict = torch.load(TENSORDICT_PATH, weights_only=False)
		print("Loaded existing tensordict.")
	else:
		tensordict = build_tensordict()
		print("Created and saved new tensordict.")

	# 2. Load or train model
	checkpoint = get_latest_checkpoint()
	
	if checkpoint:
		autoencoder = MultiTaskEncoder.load_from_checkpoint(
			checkpoint, 
			encoder=Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM),
			decoder=Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE),
			attr_heads=AttributeHead(MODEL_DIM, HIDDEN_ATTR_SIZE, OUTPUT_ATTR_SIZE),
			learning_rate=1e-3,
			alpha=0.9,
			learning_rate_w=5e-3
		)
		print(f"Loaded model from checkpoint: {checkpoint}")
	else:
		autoencoder = MultiTaskEncoder(
			Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM), 
			Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE),
			AttributeHead(MODEL_DIM, HIDDEN_ATTR_SIZE, OUTPUT_ATTR_SIZE),
			learning_rate=1e-3,
			alpha=0.9,
			learning_rate_w=5e-3
		)
		print(f"No checkpoint found. Training initialized model for {EPOCHS} epochs...")
		# For training, need a DataLoader

	trainer = Trainer(max_epochs=EPOCHS)
	train_loader = torch.utils.data.DataLoader(
		tensordict,
		batch_size=BATCH_SIZE,
		shuffle=True,
		collate_fn=lambda x: x 
	)
	trainer.fit(model=autoencoder, train_dataloaders=train_loader)
	# Save checkpoint manually if needed
	os.makedirs("models", exist_ok=True)
	ckpt_path = "models/autoencoder_trained.ckpt"
	trainer.save_checkpoint(ckpt_path, weights_only=False)
	print(f"Model trained and saved to {ckpt_path}")

if __name__ == "__main__":
	main()
