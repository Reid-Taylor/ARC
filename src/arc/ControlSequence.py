# Imports
import os
import glob
import torch
import numpy as np
from tensordict import TensorDict
from lightning.pytorch import Trainer
from OutputGrid import LitAutoEncoder, Encoder, Decoder, positional_encodings
from ARCDataClasses import ARCProblemSet

GRID_SIZE = 30*30
ATTENTION_SIZES = (32, 128, 512)
MODEL_DIM = 64
TENSORDICT_PATH = "tensordict_training.pt"
CHECKPOINT_DIR = "../../lightning_logs/"

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
				"reconstructed_grid": None
			},
			batch_size=len(batch)
		)
	# Save the tensordict for future use
	tensordict = collate_fn(training_dataset)
	torch.save(tensordict, TENSORDICT_PATH)
	return tensordict

def main():
	if os.path.exists(TENSORDICT_PATH):
		tensordict = torch.load(TENSORDICT_PATH)
		print("Loaded existing tensordict.")
	else:
		tensordict = build_tensordict()
		print("Created and saved new tensordict.")

	# 2. Load or train model
	checkpoint = get_latest_checkpoint()
	autoencoder = LitAutoEncoder(Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM), Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE))
	if checkpoint:
		autoencoder = autoencoder.load_from_checkpoint(checkpoint)
		
		print(f"Loaded model from checkpoint: {checkpoint}")
	else:
		print("No checkpoint found. Training model for 50 epochs...")
		trainer = Trainer(max_epochs=50)
		# For training, need a DataLoader
		train_loader = torch.utils.data.DataLoader(
			tensordict,
			batch_size=23,
			shuffle=True,
			collate_fn=lambda x: x 
		)
		trainer.fit(model=autoencoder, train_dataloaders=train_loader)
		# Save checkpoint manually if needed
		ckpt_path = "autoencoder_trained.ckpt"
		trainer.save_checkpoint(ckpt_path)
		print(f"Model trained and saved to {ckpt_path}")

if __name__ == "__main__":
	main()
