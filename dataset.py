import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


class XORDataset:
    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise
        """
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Generate random binary data points
        data = np.random.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.int64)  # XOR label
        # Add Gaussian noise to the data points
        data += self.std * np.random.randn(*data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data points we have
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    plt.show()


def visualize_classification(model, data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    # Create a grid for predictions
    x1 = np.arange(-0.5, 1.5, step=0.01)
    x2 = np.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = np.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = np.stack([xx1, xx2], axis=-1)

    # Get predictions from the model
    preds = model.forward(
        model_inputs.reshape(-1, 2)
    )  # Assuming model has a predict method
    preds = preds.reshape(xx1.shape)

    # Clip predictions to the range [0, 1]
    preds = np.clip(preds, 0, 1)

    c0 = np.array(to_rgba("C0"))
    c1 = np.array(to_rgba("C1"))
    output_image = (1 - preds[:, :, None]) * c0 + preds[
        :, :, None
    ] * c1  # Color blending
    plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    plt.show()
