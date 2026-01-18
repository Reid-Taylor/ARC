from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from src.arc.ARCNetworks import TransformationSpaceProjection
from src.arc.ARCDataClasses import ARCProblemSet
from itertools import combinations

@beartype
class TransformationDescriber(L.LightningModule):
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
            learning_rate:float=1e-3, 
            alpha:float=0.85,
            **network_dimensions
        ) -> None:
        """
        This network should receive as inputs the embeddings of the example challenges and solutions. We can use contrastive learning approaches to create a "transformation" space, which is trained to create similar embeddings representing the relationships between example challenges and solutions. The relationships for examples under the same name must be encoded as near-identical--Ultimately, we will take the average of these embeddings of the many examples to provide the "transformation" description provided at test time. We can prompt the network to learn more quickly the desirable characteristics by leveraging data augmentations (of sensitive tasks, that is) to various input & (constructed) output grids to enforce the learning of transformations. We can train this first piece to explicitly learn what should encoded as the null-transformation, the identity function.
        """
        super().__init__()
        # self.transformation_description = TensorDictModule(
        #     TransformationSpaceProjection(
        #         **network_dimensions["TransformationDescriber"]
        #     ),
        #     in_keys=["example_input_embedding","example_output_embedding", "input_randomly_augmented"],
        #     out_keys=["transformation_description", "random_description"]
        # )
        self.transformation_description = TransformationSpaceProjection(
            **network_dimensions["TransformationDescriber"]
        )

        self.num_tasks = 7
        
        self.lr: float = learning_rate
        self.raw_w: Float[torch.Tensor, "A"] = torch.nn.Parameter(torch.zeros(self.num_tasks))
        self.alpha: float = alpha
        self.custom_task_weighting = torch.tensor([1,1,0.5,2,1,1,1], dtype=torch.float32)

        self.register_buffer("L0", torch.zeros(self.num_tasks))
        self.L0_initialized: bool = False

        self.automatic_optimization: bool = False

    def _get_parameters(self):
        params = [p for p in self.transformation_description.parameters() if p.requires_grad]

        return params
    
    def _task_weights(self) -> Float[torch.Tensor, "A"]:
        #TODO: consider weighting across reconstruction?
        w = F.softmax(self.raw_w, dim=0) + 1e-8
        w = self.num_tasks * w / w.sum()
        return w
    
    def forward(self, x):
        results = {
            "standard" : [],
            "backwards" : [],
            "augmentation" : [],
            "identic" : []
        }

        for idx in range(x['num_examples'].item()):
            input_example: Float[torch.Tensor, "1 D"] = x['examples'][f'example_{idx}']['input']['embedding']
            output_example: Float[torch.Tensor, "1 D"] = x['examples'][f'example_{idx}']['output']['embedding']
            input_augmentation: Float[torch.Tensor, "1 D"] = x['examples'][f'example_{idx}']['input']['augmented_grid_embedding']
            output_augmentation: Float[torch.Tensor, "1 D"] = x['examples'][f'example_{idx}']['output']['augmented_grid_embedding']

            transform_results = self.transformation_description(input_example, output_example, input_augmentation)
            example_transformation_description = transform_results["transformation"]
            example_random_description = transform_results["random"]

            backwards_results = self.transformation_description(output_example, input_example, output_augmentation)
            example_backwards_transformation_description = backwards_results["transformation"]
            example_backwards_random_description = backwards_results["random"]

            identic_input_results = self.transformation_description(input_example, input_example, output_augmentation)
            identic_input_description = identic_input_results["transformation"]
            
            identic_output_results = self.transformation_description(output_example, output_example, output_augmentation)
            identic_output_description = identic_output_results["transformation"]

            results['standard'].append(example_transformation_description)
            results["backwards"].append(example_backwards_transformation_description)
            results['augmentation'].extend([example_random_description,example_backwards_random_description])
            results["identic"].extend([identic_input_description,identic_output_description])

        challenge_input: Float[torch.Tensor, "1 D"] = x['challenge']['embedding']
        challenge_output: Float[torch.Tensor, "1 D"] = x['solution']['embedding']
        input_augmentation: Float[torch.Tensor, "1 D"] = x['examples'][f'example_{idx}']['input']['augmented_grid_embedding']
        output_augmentation: Float[torch.Tensor, "1 D"] = x['examples'][f'example_{idx}']['output']['augmented_grid_embedding']

        challenge_transform_results = self.transformation_description(challenge_input, challenge_output, input_augmentation)
        example_transformation_description = challenge_transform_results["transformation"]
        example_random_description = challenge_transform_results["random"]

        challenge_backwards_results = self.transformation_description(challenge_output, challenge_input, output_augmentation)
        example_backwards_transformation_description = challenge_backwards_results["transformation"]
        example_backwards_random_description = challenge_backwards_results["random"]

        challenge_identic_input_results = self.transformation_description(challenge_input, challenge_input, output_augmentation)
        identic_input_description = challenge_identic_input_results["transformation"]
        
        challenge_identic_output_results = self.transformation_description(challenge_output, challenge_output, output_augmentation)
        identic_output_description = challenge_identic_output_results["transformation"]

        results['standard'].append(example_transformation_description)
        results["backwards"].append(example_backwards_transformation_description)
        results['augmentation'].extend([example_random_description,example_backwards_random_description])
        results["identic"].extend([identic_input_description,identic_output_description])

        return results
    
    def training_step(self, batch):
        opt_model, opt_w = self.optimizers()
        results = self.forward(batch)
        all_params = self._get_parameters()
        
        high_proximity_loss = 0
        low_proximity_loss = 0
        opposite_task_loss = 0
        identic_loss = 0

        all_embeddings = []
        for key in ['standard', 'backwards', 'augmentation']:
            all_embeddings.extend(results[key])
        all_embeddings = torch.stack(all_embeddings)

        for x,y in combinations(results['standard'], 2):
            high_proximity_loss += (
                1 - F.cosine_similarity(
                    x, 
                    y,
                    dim=-1
                ).mean()
            )

        for x,y in combinations(results['backwards'], 2):
            high_proximity_loss += (
                1 - F.cosine_similarity(
                    x, 
                    y,
                    dim=-1
                ).mean()
            )
        
        for x in results['standard']:
            for y in results['backwards']:
                opposite_task_loss += (
                    1 + F.cosine_similarity(
                        x, 
                        y,
                        dim=-1
                    ).mean()
                )

        for x in results['backwards']:
            for y in results['augmentation']:
                low_proximity_loss += (
                    F.cosine_similarity(
                        x, 
                        y,
                        dim=-1
                    ).mean().abs()
                )

        for x in results['standard']:
            for y in results['augmentation']:
                low_proximity_loss += (
                    F.cosine_similarity(
                        x, 
                        y,
                        dim=-1
                    ).mean().abs()
                )

        for x in results['identic']:
            identic_loss += (
                F.mse_loss(
                    x,
                    torch.zeros_like(x),
                    dim=-1
                )
            )

        def entropy_density_loss(embeddings: torch.Tensor, lambda_entropy: float = 0.01) -> torch.Tensor:
            """Encourage high entropy in embedding magnitudes"""
            # Normalize to probability distribution
            probs = F.softmax(torch.abs(embeddings), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return lambda_entropy * torch.mean(-entropy)  # Negative to encourage high entropy
        
        def variance_density_loss(embeddings: torch.Tensor, lambda_var: float = 0.01) -> torch.Tensor:
            """Encourage high variance to prevent mode collapse"""
            variance = torch.var(embeddings, dim=-1)
            return lambda_var * torch.mean(-variance)

        def anti_sparsity_loss(embeddings: torch.Tensor, threshold: float = 0.1, lambda_sparse: float = 0.01) -> torch.Tensor:
            """Penalize activations below threshold"""
            sparse_penalty = torch.mean(torch.relu(threshold - torch.abs(embeddings)))
            return lambda_sparse * sparse_penalty

        loss = torch.stack(
            [high_proximity_loss] + 
            [low_proximity_loss] +
            [opposite_task_loss] +  
            [identic_loss] + 
            [
                entropy_density_loss(all_embeddings),
                variance_density_loss(all_embeddings), 
                anti_sparsity_loss(all_embeddings)
            ]
        )

        # Initialize L0 if not already done
        if (not self.L0_initialized):
            self.L0[:] = loss.detach()
            self.L0_initialized = True
            L0_for_computation = loss.detach()
        else:
            L0_for_computation = self.L0

        # Get task weights. Compute gradient norms for each task
        w = self._task_weights()
        GRADIENT_LIST = []
        loss_total = torch.sum((w * loss))

        def _get_gradient(relevant_weighted_losses) -> List[torch.Tensor]:
            gradients = torch.autograd.grad(relevant_weighted_losses, all_params, retain_graph=True, create_graph=True, allow_unused=True)
            list_of_gradients = [g for g in gradients if g is not None]

            if list_of_gradients:
                return torch.norm(torch.stack([g.norm() for g in list_of_gradients]), 2)
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        # Compute gradients per task
        for i in range(self.num_tasks):
            GRADIENT_LIST.append(
                _get_gradient(
                    w[i] * loss[i] * self.custom_task_weighting[i]
                )
            )
        
        # Compute mean gradient norm
        GRADIENT_TENSOR = torch.stack(GRADIENT_LIST)
        AVERAGE_GRADIENT = GRADIENT_TENSOR.mean()

        loss_hat = (loss.detach() / (L0_for_computation + 1e-6))
        loss_hat /= loss_hat.mean()
        target_G = AVERAGE_GRADIENT.detach() * (loss_hat ** self.alpha)
        
        loss_grad = torch.sum(torch.abs(GRADIENT_TENSOR - target_G))

        # Update parameters
        opt_model.zero_grad()
        opt_w.zero_grad()

        loss_total.backward(retain_graph=True)
        loss_grad.backward()

        opt_model.step()
        opt_w.step()

        # Log metrics and updates
        self.log_dict(
            {
                "train/total_loss": loss_total.detach(),
                "train/loss_grad": loss_grad.detach(),
            },
            prog_bar=True
        )

    def configure_optimizers(self):
        params = self._get_parameters()

        opt_model = torch.optim.Adam(
            params=params,
            lr=self.lr
        )
        opt_w = torch.optim.Adam([self.raw_w], lr=self.lr)
        return [opt_model, opt_w]