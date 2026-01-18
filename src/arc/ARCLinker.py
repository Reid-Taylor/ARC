from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, Union, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from src.arc.ARCNetworks import AttributeHead, Decoder, Encoder, FullyConnectedLayer

@beartype
class Linker(L.LightningModule):
    """
    This class should learn to predict the embeddings of an output grid given that of an output grid.

    We will use this opportunity to experiment with residual adapter modules, parameterized by a trainable function of the embedding itself, as the primary method of differentiation of the task.

    I hypothesize that by allowing the residual adapter to be a parameterization of a function of the embedding, we will achieve something similar to MAMBA-5. Assuming our list of learned-features in our embedding space is comprehensive of the information needed to describe the grid, we should see promising results.

    The base network should not be fit to any one example. We should have an end-goal network in which the network is trained to learn how to apply learnings from one grid to another. The learnings themselves of each examples unique transformations must be encapsulated wholly within the parameterization--the residual adapters. How can we achieve this?

    Enforce structure by asserting that the zero-vector of the parameterization corresponds to the identity function. Theoretically, this will force the base network to learn a generalizable mapping, while the adapters learn task-specific transformations.

    Further, we should see that "eliminating" the adapters (setting them to zero) should yield a reasonable approximation of the input grid, given the input grid. This will be a good test of the generalizability of the base network.

    Finally, we can train the network under the zero-vector adapter in a self-supervised task to enforce that the network exists only to apply a specified transformation. We might use augmentations of input grids as targets of this network, if we may learn the parameterization of these augmentation functions. (Perhaps this transition from the previous sentence to the following one is achieved through pre-training?) Then, the parameterization function must only be learned as a set of transformations describing the relationship between the challenge and solution grids.    
    """
    def __init__(
            self,
            network_dimensions: Dict[str:List[int]]
        ) -> None:
        """
        Once this "transformation" embedding is learned, we will be able to learn the network described above, which is comprised of the thus-trained embedding of transformations, and the complementary network which learns to apply such transformations. A "null" transformation must output the same embedding as inputted (through non-trivial effects of the null transformation), giving us incredible space to train various grids according to the identity function.
        """
        super().__init__()
        self.residual_adapter_module = TensorDictModule(
            Encoder(
                **network_dimensions["Encoder"]
            ),
            in_keys=["transformation_description"],
            out_keys=["network_parameters"]
        )
        self.transformation_applier = TensorDictModule(
            Encoder(
                **network_dimensions["Encoder"]
            ),
            in_keys=["network_parameters","challenge_embedding"],
            out_keys=["predicted_solution_embedding"]
        )
    
    def forward(self, x):
        pass

    def training_step(self, batch):

        if self.trainer.current_epoch % 5 == 0:
            # Apply different logic to train the overall network, while temporarily freezing the adapters
            pass
            
        # we should freeze the base network and only train the adapters 
        pass

    def configure_optimizers(self):
        pass