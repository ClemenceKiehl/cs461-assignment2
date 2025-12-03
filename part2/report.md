# Downstream evaluation on histopathology data

In this assignment, each patient is represented as a bag of patch embeddings extracted using a pretrained H&E foundation model. The downstream task is to classify cancer subtypes at the patient level, where each patient has a variable number of patches.
The baseline (“linear”) averages all patches of a patient and trains a linear classifier. However, simple averaging ignores the fact that not all patches contribute equally to diagnosis. To address this, I implemented a Multiple Instance Learning (MIL) model with attention.

The model processes each patch embedding through a small multilayer perceptron which reduces the original 3072-dimensional feature vector into a 256-dimensional hidden representation. This embedding is then fed into an attention module composed of 2 linear layers with a tanh nonlinearity in between. For each patch, the attention module outputs a scalar score, and a softmax operation normalizes all scores within a patient’s bag to obtain an attention distribution. These attention weights determine how much each patch contributes to the final patient representation. The weighted sum of the patch embeddings produces a single vector for each patient. It is then classified using another small multilayer network. I ensured that the final output of the model consists of class probabilities rather than raw logits, as the evaluation code computes multiclass ROC-AUC and therefore requires outputs that sum to one.

Furthermore, a custom collate function was necessary because the number of patches varies across patients. Instead of stacking them into a tensor, the collate function constructs a list of bags, where each bag is a tensor of shape (number_of_patches, 3072) and corresponds to a single patient. This list is passed to the model along with the associated labels, which satisfies the interface expected by the autograder. The model was trained for 20 epochs using the Adam optimizer with a learning rate of 1e-4. I used an 80-20 train–validation split and saved the checkpoint that achieved the highest validation macro-F1.

The final evaluation demonstrated a substantial improvement over the baseline:

Evaluation Results:
+-----------------+------------+------------+---------------------+-----------+
| Method          |   accuracy |   f1_score |   balanced_accuracy |   roc_auc |
+=================+============+============+=====================+===========+
| linear_baseline |     0.8083 |     0.7324 |              0.7762 |    0.935  |
+-----------------+------------+------------+---------------------+-----------+
| submission      |     0.872  |     0.8654 |              0.8775 |    0.9864 |
+-----------------+------------+------------+---------------------+-----------+
While the linear probe achieves a macro-F1 of 0.7324, my MIL model reaches 0.8654, with an accuracy of 0.8720 and a ROC-AUC of 0.9864. These results confirm that attention successfully identifies the most informative patches and that handling patients as variable-sized bags rather than averaged vectors significantly enhances performance. 

## Reproduction of Results :

To train the model, navigate to part2 and run:

``` python -m part2.models.submission ```

This creates ckpts/best_mil.pt.

To evaluate the trained model and reproduce the reported metrics, run:

``` python -m part2.eval submission ```

