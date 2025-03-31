import numpy as np
import matplotlib.pyplot as plt
from dataset import XORDataset, visualize_classification
from simple_nn import SimpleClassifier, GradientDescent


def train_model(model, data_loader, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        for data_inputs, data_labels in data_loader:
            preds = model.forward(data_inputs)
            preds = 1 / (1 + np.exp(-preds.squeeze(axis=1)))  # Sigmoid activation
            dL_dW1, dL_db1, dL_dW2, dL_db2 = model.backprop(
                data_inputs, data_labels, preds
            )

            optimizer.step(model.linear1, dL_dW1, dL_db1)
            optimizer.step(model.linear2, dL_dW2, dL_db2)


def eval_model(model, data_loader):
    correct_preds, total_preds = 0, 0

    for data_inputs, data_labels in data_loader:
        preds = model.forward(data_inputs)
        preds = 1 / (1 + np.exp(-preds.squeeze(axis=1)))  # Sigmoid activation
        pred_labels = (preds >= 0.5).astype(int)

        correct_preds += np.sum(pred_labels == data_labels)
        total_preds += data_labels.shape[0]

    print(f"Model Accuracy: {100.0 * correct_preds / total_preds:.2f}%")


def create_data_loader(dataset, batch_size=128):
    return [
        (dataset.data[i : i + batch_size], dataset.label[i : i + batch_size])
        for i in range(0, len(dataset.data), batch_size)
    ]


if __name__ == "__main__":
    train_dataset = XORDataset(size=2500)
    test_dataset = XORDataset(size=500)

    train_data_loader = create_data_loader(train_dataset)
    test_data_loader = create_data_loader(test_dataset)

    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    optimizer = GradientDescent(lr=0.01)

    visualize_classification(model, test_dataset.data, test_dataset.label)

    train_model(model, train_data_loader, optimizer)
    eval_model(model, test_data_loader)

    visualize_classification(model, test_dataset.data, test_dataset.label)
    plt.show()
