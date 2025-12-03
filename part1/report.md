# Test-time adaptation on corrupted CIFAR-10

Deep neural networks trained on clean datasets usually perform poorly when evaluated on corrupted or out-of-distribution inputs due to dataset shift. Test-Time Adaptation (TTA) addresses this issue by modifying the model during inference without requiring target labels. In this assignment, I implement a TTA strategy for a ResNet-50 pretrained on CIFAR-10 and evaluate it on a benchmark containing six corruption types. The goal is to improve performance over the Unadapted baseline and the TestTimeNorm reference implementation.

My method is based on the TENT algorithm. Instead of updating all network parameters, I adapted only the affine parameters (γ and β) of all BatchNorm layers. These parameters directly influence feature normalization and are known to be effective for addressing distribution shift. This approach avoids modifying convolutional weights, making adaptation stable and lightweight.

The implementation resides in tta/submission.py within the Submission class. Upon initialization, all model parameters are frozen, and only BatchNorm affine parameters are marked as trainable. An Adam optimizer is created over these parameters. For each incoming test batch, the model is switched to training mode so that gradients can be computed. I computed the softmax distribution and minimized the mean entropy of the predictions. This helps the model becoming more confident on the current target-domain inputs. The optimization typically runs for a small number of steps per batch (here, one step), which keeps inference fast while still providing meaningful adaptation. The forward pass then returns the logits from the adapted model.
Because the evaluation benchmark is organized by corruption type, adaptation must not leak from one corruption scenario to another. To ensure this, I saved the full pretrained model state at initialization. Before each corruption subset is evaluated, the reset() method restores all model weights to their original state and reinitializes the optimizer. This ensures an independent evaluation across corruptions.

Running python -m eval submission yields the following results

| Scenario      | Unadapted | TestTimeNorm | Submission |
|---------------|-----------|--------------|------------|
| contrast      | 0.6684    | 0.8565       | 0.8654     |
| fog           | 0.8408    | 0.8723       | 0.8827     |
| frost         | 0.8022    | 0.8212       | 0.8403     |
| gaussian_blur | 0.7354    | 0.8810       | 0.8876     |
| pixelate      | 0.8324    | 0.8609       | 0.8716     |
| shot_noise    | 0.5682    | 0.7687       | 0.7967     |
| **Mean Accuracy** | **0.7412** | **0.8434** | **0.8574** |

The TENT-based submission method outperforms both baselines, achieving a mean accuracy of 0.857, the strongest performance among the compared methods.

## Reproduction of results

To reproduce the results on Gnoto:

``` cd ~/cs461-assignment2/part1 ```

``` python -m eval submission ```
