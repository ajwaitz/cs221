#!/usr/bin/python
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from einops import reduce, rearrange, einsum

from util import Vocabulary, read_tweet_data

# number of output prediction classes in problem 3 and 4
K = 3

############################################################
# Problem 3: NumPy linear classifier
############################################################

############################################################
# Problem 3a: Build vocabulary


def build_vocabulary(examples: List[str]) -> Vocabulary:
    """
    Build vocabulary from examples using the Vocabulary() class in util.py.

    @param examples: List of text strings
    @return: a Vocabulary() object
    """
    vocab = Vocabulary()

    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    # return vocab
    for example in examples:
        tokens = example.split(" ")
        for token in tokens:
            vocab.add_word(token)
    return vocab
    # END_YOUR_CODE


############################################################
# Problem 3b: Text to features


def text_to_features(text, vocab) -> np.ndarray:
    """
    Convert a given text string to a sparse feature representation.

    @param text: An input text string to be converted to features
    @param vocab: A Vocabulary() object containing the word-to-index mapping

    @return: A numpy array of shape (vocab.size(),) where each element represents
        the count of the corresponding vocabulary word in the input text. Words not
        in vocabulary are mapped to the <UNK> token index.
    """
    features = np.zeros(vocab.size())

    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    # return features
    if text == "":
        return features
    tokens = text.split(" ")
    for token in tokens:
        index = vocab.get_index(token)
        features[index] += 1
    return features
    # END_YOUR_CODE


############################################################
# Problem 3c: Numpy softmax


def numpy_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Use NumPy library functions to compute softmax probabilities for a given batch of logits.

    The softmax function converts a vector of real numbers into a probability distribution.
    Mathematical formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

    For numerical stability, we use the equivalent formula:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    @param logits: 2D array of shape (batch_size, k) containing raw scores
    @return: A 2D array of shape (batch_size, k) with softmax probabilities.
             Each row sums to 1.0 and all values are in range [0, 1].

    Example:
        Input:  [[2.0, 1.0, 0.1]]  (raw logits for 3 classes)
        Output: [[0.659, 0.242, 0.099]]  (probabilities summing to 1.0)
    """
    logits = logits - reduce(logits, "batch class -> batch 1", "max")

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # return None
    e = np.exp(logits)
    s = np.sum(e, axis=1, keepdims=True)
    return e / s
    # END_YOUR_CODE


############################################################
# Problem 3d: NumPy cross entropy loss


def numpy_cross_entropy_loss(predictions: np.ndarray,
                             targets: np.ndarray,
                             epsilon: float = 1e-15) -> float:
    """
    Use NumPy library functions to compute cross-entropy loss between predictions and one-hot targets.

    @param predictions: softmax probabilities in a 2D array of shape (batch_size, K)
    @param targets: one-hot encoded labels in a 2D array of shape (batch_size, K)
    @param epsilon: A small number to add to predictions before taking the logarithm,
        to prevent undefined behavior when taking log(0)

    @return: Average cross-entropy loss (scalar value)
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # return None
    # scores = (predictions * targets).sum(axis=-1)
    return -np.sum(targets * np.log(predictions + epsilon))
    # END_YOUR_CODE


############################################################
# Problem 3e: NumPy compute gradients


def numpy_compute_gradients(features: np.ndarray, predictions: np.ndarray,
                            targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients (the partial derivatives of the cross-entropy loss)
    for the weights and bias using batch gradient descent. Batch gradient descent
    averages gradients across all examples in the batch,
    providing a more stable estimate than single-example gradients.

    @param features: 2D array of shape (batch_size, num_features) - input features
    @param predictions: 2D array of shape (batch_size, K) - softmax probabilities
    @param targets: 2D array of shape (batch_size, K) - one-hot encoded labels

    @return: A tuple of (grad_weights, grad_bias)
        - grad_weights: shape (num_features, K) - gradients w.r.t. weights
        - grad_bias: shape (1, K) - gradients w.r.t. bias
    """
    batch_size = features.shape[0]

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # return None
    d_weights = einsum(features, targets * (predictions - 1), "b f, b k -> b f k")
    d_bias = targets * (predictions - 1)
    out = (d_weights.mean(axis=0), d_bias.mean(axis=0, keepdims=True))
    return out
    # END_YOUR_CODE


############################################################
# Problem 3f: Linear classifier predictor


def predict_linear_classifier(features: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                                bias: np.ndarray) -> float:
    """
    Make predictions using the trained linear classifier weights and bias and compute the accuracy
    of the predictions by comparing them to the true labels.

    @param features: NumPy array of features of dimensions (num_examples, num_features)
    @param labels: NumPy array of true labels of dimensions (num_examples, K) in one-hot format
    @param weights: Trained weights of shape (num_features, K)
    @param bias: Trained bias of shape (1, K)

    @return: Accuracy as a float
    """
    # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
    # return 0.0
    scores = numpy_softmax(features @ weights + bias)
    predictions = np.zeros_like(scores)
    np.put_along_axis(arr=predictions, indices=np.expand_dims(np.argmax(scores, axis=1), axis=1), values=1, axis=1)
    check = (predictions == labels).all(axis=1)
    return check.astype(int).mean().item()
    # END_YOUR_CODE


############################################################
# Problem 3g: Train linear classifier


def train_linear_classifier(train_features: np.ndarray, train_labels: np.ndarray,
                                         val_features: np.ndarray, val_labels: np.ndarray,
                                         num_epochs: int, lr: float) -> np.ndarray:
    """
    Train a linear classifier using NumPy arrays and full-batch gradient descent.
    At the end of each epoch, call your predictor function to evaluate your trained parameters
    on the validation data and print out training loss and validation accuracy.

    @param train_features: NumPy array of training features of dimensions (num_train_examples, num_features)
    @param train_labels: NumPy array of training labels of dimensions (num_train_examples, K)
    @param val_features: NumPy array of validation features of dimensions (num_val_examples, num_features)
    @param val_labels: NumPy array of validation labels of dimensions (num_val_examples, K)
    @param num_epochs: number of training epochs
    @param lr: learning rate

    @return: Trained weights in a NumPy array of shape (num_features, K), and of shape (1, K)

    You should use the numpy_softmax, numpy_cross_entropy_loss, numpy_compute_gradients,
    and predict_linear_classifier functions you implemented.
    """

    num_train_examples, num_features = train_features.shape

    weights = np.random.randn(num_features, K)
    bias = np.zeros((1, K))

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    # return weights, bias
    for i in range(num_epochs):
        scores = numpy_softmax(train_features @ weights + bias)

        d_weight, d_bias = numpy_compute_gradients(train_features, scores, train_labels)

        print("weight mean", weights.mean(), d_weight.mean())
        print("bias mean", bias.mean(), d_bias.mean())

        weights = weights - lr * d_weight
        bias = bias - lr * d_bias

        train_acc = predict_linear_classifier(train_features, train_labels, weights, bias) * 100
        val_acc   = predict_linear_classifier(val_features, val_labels, weights, bias) * 100

        train_loss = numpy_cross_entropy_loss(scores, train_labels)
        val_loss = numpy_cross_entropy_loss(numpy_softmax(val_features @ weights + bias), val_labels)

        print(f"Epoch {i}. Train acc {train_acc:.2f}. Val acc {val_acc:.2f}")
        print(f"           Train loss {train_loss:.2f}. Val loss {val_loss:.2f}")

    return (weights, bias)
    # END_YOUR_CODE


############################################################
# Problem 4: Multilayer Perceptron (MLP) with embeddings
############################################################

############################################################
# Problem 4a: Text to average embedding


def text_to_average_embedding(text: str, vocab: Vocabulary,
                              embedding_layer: nn.Embedding) -> torch.Tensor:
    """
    Convert text to an averaged embedding vector using learnable embeddings.

    @param text: Input text string
    @param vocab: Vocabulary object
    @param embedding_layer: PyTorch embedding layer
    @return: A single tensor representing the averaged embedding
    """
    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    # return None
    tokens = text.split(" ")
    n = 0
    average_embedding = None
    for token in tokens:
        embedding = vocab.get_vector(token, embedding_layer)
        if average_embedding is None:
            average_embedding = embedding.clone().detach()
        else:
            average_embedding += embedding
        n += 1

    return average_embedding / n
    # END_YOUR_CODE


############################################################
# Problem 4b: Extract averaged features


def extract_averaged_features(texts: List[str], vocab: Vocabulary,
                              embedding_layer: nn.Embedding) -> torch.Tensor:
    """
    Extract averaged embedding features for all texts.

    @param texts: List of text strings
    @param vocab: Vocabulary object
    @param embedding_layer: PyTorch embedding layer
    @return: Tensor of shape (num_texts, embedding_dim)
    """
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    # return None
    all_embeds = []
    for i, text in enumerate(texts):
        all_embeds.append(text_to_average_embedding(text, vocab, embedding_layer))
    out = torch.stack(all_embeds, dim=0)
    return out
    # END_YOUR_CODE


############################################################
# Problem 4c: MLP Classifier


class MLPClassifier(nn.Module):
    """
    A simple neural network that uses averaged embeddings for text classification.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # We initialize the layer weights with Xavier uniform, which helps mitigate
        # vanishing or exploding gradients. If you are curious, you can read more here:
        # https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        @param x: Input tensor of averaged embeddings, shape (batch_size, embedding_dim)
        @return: Raw logits of shape (batch_size, output_dim)
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        # return x
        return self.fc2(self.relu(self.fc1(x)))
        # END_YOUR_CODE


############################################################
# Problem 4d: Utility functions


def torch_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Use PyTorch library functions to compute softmax probabilities from logits.

    @param logits: PyTorch tensor of shape (batch_size, K) containing raw logits
    @return: PyTorch tensor of shape (batch_size, K) where each row sums to 1.0
    and all values are probabilities in range [0, 1]
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    # return None
    return nn.functional.softmax(logits, dim=-1)
    # END_YOUR_CODE


def torch_cross_entropy_loss(predictions: torch.Tensor,
                             targets: torch.Tensor,
                             epsilon: float = 1e-15) -> torch.Tensor:
    """
    Compute cross-entropy loss between predictions and one-hot targets. 
    Your output should be the mean loss of the entire batch.

    @param predictions: PyTorch tensor of shape (batch_size, K) containing
        softmax probabilities for each class (values in [0,1], each row sums to 1.0)
    @param targets: PyTorch tensor of shape (batch_size, K) containing
        one-hot encoded true labels (each row has exactly one 1 and rest are 0s)
    @param epsilon: Small floating point value (default 1e-15) added to predictions
        before taking logarithm to prevent numerical instability from log(0)
    @return: Scalar PyTorch tensor containing the mean cross-entropy loss across the batch
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    # return None
    return nn.functional.cross_entropy(predictions + epsilon, targets.float())
    # END_YOUR_CODE


def update_parameter(param: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    """
    Manually update parameters in place using gradient (stochastic gradient descent).

    @param param: The parameter tensor to update (modified in-place)
    @param grad: Gradient tensor of the same shape as param, containing partial derivatives
        of the loss function with respect to the parameter
    @param lr: Learning rate (scalar float) controlling the step size of parameter updates
    @return: None (parameters are updated in-place)

    Hint: Don't forget to wrap your code in `with torch.no_grad():`, to temporarily disable
    gradient calculation.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # pass
    with torch.no_grad():
        param -= lr * grad
    # END_YOUR_CODE


############################################################
# Problem 4e: MLP Predictor


def predict_mlp(texts: List[str], labels: torch.Tensor, classifier: nn.Module,
                embedding_layer: nn.Embedding, vocab: Vocabulary) -> float:
    """
    Make predictions on new texts using the trained embedding-based classifier
    and computes the accuracy of the predictions by comparing them to the true labels.

    @param texts: List of text strings to classify
    @param labels: One-hot tensor of dimension (len(texts), K) encoding labels of the texts
    @param classifier: Trained classifier model
    @param embedding_layer: Trained embedding layer
    @param vocab: Vocabulary object
    @return: Accuracy of the model's predictions

    Hint: Don't forget to wrap your code in `with torch.no_grad():`, to temporarily disable
    gradient calculation.
    """
    classifier.eval()
    embedding_layer.eval()

    accuracy = 0.0

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    # return accuracy
    inputs = extract_averaged_features(texts, vocab, embedding_layer)
    with torch.no_grad():
        outputs = classifier.forward(inputs)

    outputs = torch.argmax(outputs, dim=-1)
    _labels = torch.argmax(labels, dim=-1)

    accuracy = (outputs == _labels).float().mean()

    return accuracy.item()
    # END_YOUR_CODE


############################################################
# Problem 4f: Train embedding MLP classifier


def train_mlp_classifier(
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        num_epochs: int,
        lr: float,
        embedding_dim: int,
        hidden_dim: int,
        batch_size: int,
        num_classes: int = K) -> Tuple[nn.Module, nn.Embedding, Vocabulary]:
    """
    Train a text classifier using learnable word embeddings. At the end of each epoch,
    call your predictor function to evaluate your trained parameters on the validation data
    and print out training loss and validation accuracy.

    @param train_texts: List of training text strings
    @param train_labels: Numpy array of training labels (one-hot encoded)
    @param val_texts: List of validation text strings
    @param val_labels: Validation labels (one-hot encoded)
    @param num_epochs: Number of training epochs
    @param lr: Learning rate for optimization
    @param embedding_dim: Dimension of word embeddings
    @param hidden_dim: Hidden dimension of classifier
    @param batch_size: Batch size for training
    @param num_classes: Number of output classes
    @return: Tuple of (trained classifier, trained embeddings, vocabulary)

    Hint 1: Don't forget to switch between training and eval mode for your classifier
    and embeddings with .train() and .eval().
    Hint 2: Use loss.backward() to compute the backward pass.

    You should be calling the helper functions you previously implemented.
    """
    vocab = build_vocabulary(train_texts)

    # Create embedding layer
    embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)
    nn.init.xavier_uniform_(embedding_layer.weight)

    # BEGIN_YOUR_CODE (our solution is 35 lines of code, but don't worry if you deviate from this)
    # pass
    import math

    train_labels = torch.from_numpy(train_labels)
    val_labels = torch.from_numpy(val_labels)

    model = MLPClassifier(embedding_dim, hidden_dim, num_classes)

    train_data = extract_averaged_features(train_texts, vocab, embedding_layer)

    num_batches = math.ceil(train_data.shape[0] / batch_size)

    for i in range(num_epochs):
        avg_loss = 0
        model.train()
        embedding_layer.train()

        # shuffle
        train_data = train_data[torch.randperm(train_data.shape[0])]

        for b in range(num_batches):
            b_start = b * batch_size
            b_end = (b+1) * batch_size
            train_batch = train_data[b_start : b_end]
            out = torch_softmax(model.forward(train_batch))

            loss = torch_cross_entropy_loss(out, train_labels[b_start : b_end])
            avg_loss += loss.detach()
            loss.backward()

            for param in model.parameters():
                update_parameter(param, param.grad, lr)
                param.grad.zero_()

        avg_loss /= num_batches
        model.eval()
        embedding_layer.eval()
        with torch.no_grad():
            val_acc = predict_mlp(val_texts, val_labels, model, embedding_layer, vocab)
            train_acc = predict_mlp(train_texts, train_labels, model, embedding_layer, vocab)
        
        print(f"Epoch {i}. Avg loss {avg_loss}. Val acc {val_acc:.2f}. Train acc {train_acc:.2f}")
        
    return (model, embedding_layer, vocab)
    # END_YOUR_CODE


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train sentiment classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples: python submission.py --model mlp --lr 0.001""")

    parser.add_argument(
        '--model',
        type=str,
        choices=['linear', 'mlp'],
        required=True,
        help='Model type to train: linear (Linear Classifier) or mlp (Multi-Layer Perceptron)')
    parser.add_argument('--lr',
                        type=float,
                        help='Learning rate (float value, e.g., 0.001, 0.01, 0.1)')
    args = parser.parse_args()

    train_texts, train_labels = read_tweet_data('tweet.train')
    val_texts, val_labels = read_tweet_data('tweet.test')

    if args.model == 'linear':
        lr = 0.2 if not args.lr else args.lr
        print(f"Training linear classifier with learning rate {lr}...")
        vocab = build_vocabulary(train_texts)

        train_features = np.array([text_to_features(text, vocab) for text in train_texts])
        val_features = np.array([text_to_features(text, vocab) for text in val_texts])

        weights, bias = train_linear_classifier(train_features,
                                                             train_labels,
                                                             val_features,
                                                             val_labels,
                                                             num_epochs=15,
                                                             lr=lr)

        accuracy = predict_linear_classifier(val_features, val_labels, weights, bias)

        print(f"NumPy classifier test accuracy: {accuracy:.4f}")

    else:
        lr = 3e-6 if not args.lr else args.lr
        print(f"Training PyTorch classifier with learning rate {lr}...")
        classifier, embedding_layer, vocab = train_mlp_classifier(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            num_epochs=15,
            lr=lr,
            embedding_dim=8,
            hidden_dim=16,
            batch_size=16,
            num_classes=train_labels.shape[1])

        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
        accuracy = predict_mlp(val_texts, val_labels_tensor, classifier, embedding_layer, vocab)

        print(f"PyTorch embedding classifier test accuracy: {accuracy:.4f}")
