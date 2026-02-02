from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from jaxtyping import Float
from src.arc.ARCNetworks import TransformationSpaceProjection
from src.arc.ARCUtils import entropy_density_loss, variance_density_loss, anti_sparsity_loss
from itertools import combinations

@beartype
class TransformationDescriber(L.LightningModule):
    """
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
        self.transformation_description = TransformationSpaceProjection(
            **network_dimensions["TransformationDescriber"]
        )

        self.num_tasks = 7
        
        self.lr: float = learning_rate
        self.alpha: float = alpha

        self.automatic_optimization: bool = False

    def _get_parameters(self):
        params = [p for p in self.transformation_description.parameters() if p.requires_grad]

        return params
    
    def forward(self, x):
        results = {
            "standard" : [],
            "backwards" : [],
            "identic" : []
        }

        for idx in range(len(x['inputs'])):
            input_example: Float[torch.Tensor, "1 D"] = x['inputs'][idx]
            output_example: Float[torch.Tensor, "1 D"] = x['outputs'][idx]

            example_transformation_description = self.transformation_description(input_example, output_example)

            example_backwards_transformation_description = self.transformation_description(output_example, input_example)

            identic_input_description = self.transformation_description(input_example, input_example)
            
            identic_output_description = self.transformation_description(output_example, output_example)

            results['standard'].append(example_transformation_description)
            results["backwards"].append(example_backwards_transformation_description)
            results["identic"].extend([identic_input_description,identic_output_description])

        challenge_input: Float[torch.Tensor, "1 D"] = x['challenge']
        challenge_output: Float[torch.Tensor, "1 D"] = x['solution']

        example_transformation_description = self.transformation_description(challenge_input, challenge_output)

        example_backwards_transformation_description = self.transformation_description(challenge_output, challenge_input)

        identic_input_description = self.transformation_description(challenge_input, challenge_input)
        
        identic_output_description = self.transformation_description(challenge_output, challenge_output)

        results['standard'].append(example_transformation_description)
        results["backwards"].append(example_backwards_transformation_description)
        results["identic"].extend([identic_input_description,identic_output_description])

        return results
    
    def training_step(self, batch):
        opt_model = self.optimizers()
        results = self.forward(batch)
        all_params = self._get_parameters()
        
        high_proximity_loss = 0
        opposite_task_loss = 0
        identic_loss = 0

        all_embeddings = []
        for key in ['standard', 'backwards']:
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

        for x in results['identic']:
            identic_loss += (
                F.mse_loss(
                    x,
                    torch.zeros_like(x)
                )
            )

        entropy_density_loss = entropy_density_loss(all_embeddings)
        variance_density_loss = variance_density_loss(all_embeddings)
        anti_sparsity_loss = anti_sparsity_loss(all_embeddings)

        loss = torch.stack(
            [high_proximity_loss] + 
            [opposite_task_loss] +  
            [identic_loss] + 
            [
                entropy_density_loss,
                variance_density_loss, 
                anti_sparsity_loss
            ]
        )

        loss_total = torch.sum(loss)

        def _get_task_gradients() -> Dict[int, torch.Tensor]:
            task_gradients = {}
            shared_params = all_params
            
            for task_idx in range(self.num_tasks):
                opt_model.zero_grad()
                
                task_loss = loss[task_idx]
                task_grads = torch.autograd.grad(
                    task_loss, 
                    shared_params, 
                    retain_graph=True, 
                    create_graph=False,
                    allow_unused=True
                )
                
                flattened_grads = []
                for grad in task_grads:
                    if grad is not None:
                        flattened_grads.append(grad.flatten())
                    else:
                        flattened_grads.append(torch.zeros(1, device=loss.device))
                
                task_gradients[task_idx] = torch.cat(flattened_grads)
            
            return task_gradients

        def _apply_pcgrad(task_gradients: Dict[int, torch.Tensor]) -> torch.Tensor:
            """Apply PCGrad algorithm to resolve gradient conflicts."""
            num_tasks = len(task_gradients)
            projected_gradients = {i: task_gradients[i].clone() for i in range(num_tasks)}
            
            conflicts = 0
            total_comparisons = 0
            
            for i in range(num_tasks):
                for j in torch.randperm(num_tasks):
                    if i != j:
                        g_i = projected_gradients[i]
                        g_j = task_gradients[j.item()]
                        
                        dot_product = torch.dot(g_i, g_j)
                        norm_i = torch.norm(g_i, p=2)
                        norm_j = torch.norm(g_j, p=2)
                        
                        total_comparisons += 1
                        if norm_i > 0 and norm_j > 0:
                            cosine_sim = dot_product / (norm_i * norm_j)
                            
                            if cosine_sim < 0:
                                conflicts += 1
                                projection = (dot_product / (norm_j ** 2)) * g_j
                                projected_gradients[i] = g_i - projection
            
            if total_comparisons > 0:
                global conflict_ratio
                conflict_ratio = conflicts / total_comparisons
            
            final_gradient = torch.stack(list(projected_gradients.values())).mean(dim=0)
            return final_gradient

        def _apply_gradients_to_params(final_gradient: torch.Tensor):
            shared_params = all_params
            param_shapes = [p.shape for p in shared_params]
            param_sizes = [p.numel() for p in shared_params]
            
            start_idx = 0
            for param, size, shape in zip(shared_params, param_sizes, param_shapes):
                param_grad = final_gradient[start_idx:start_idx + size].reshape(shape)
                param.grad = param_grad
                start_idx += size

        task_gradients = _get_task_gradients()
        final_gradient = _apply_pcgrad(task_gradients)
        
        opt_model.zero_grad()
        loss_total.backward()
        
        _apply_gradients_to_params(final_gradient)
        
        torch.nn.utils.clip_grad_norm_(
            [p for params_list in all_params.values() if isinstance(params_list, list) for p in params_list] +
            [p for params_dict in all_params.values() if isinstance(params_dict, dict) for params_list in params_dict.values() for p in params_list],
            max_norm=1.0
        )

        opt_model.step()

        self.log_dict(
            {
                "train/total_loss": loss_total.detach(),
                "train/high_proximity_loss": high_proximity_loss,
                "train/opposite_task_loss": opposite_task_loss,  
                "train/identic_loss": identic_loss, 
                "train/entropy_density_loss": entropy_density_loss,
                "train/variance_density_loss": variance_density_loss, 
                "train/anti_sparsity_loss": anti_sparsity_loss,
                "train/conflict_ratio": conflict_ratio
            },
            prog_bar=True
        )

    def configure_optimizers(self):
        params = self._get_parameters()

        opt_model = torch.optim.Adam(
            params=params,
            lr=self.lr
        )

        return opt_model