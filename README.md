```python
%matplotlib inline
from dataset import visualize_classification, XORDataset
import matplotlib.pyplot as plt
from simple_nn import SimpleClassifier, GradientDescent
from train_nn import train_model, eval_model, create_data_loader
```


```python

num_inputs = 2
num_hidden = 4
num_outputs = 1

train_dataset = XORDataset(size=2500)
test_dataset = XORDataset(size=500)

train_data_loader = create_data_loader(train_dataset)
test_data_loader = create_data_loader(test_dataset)


model = SimpleClassifier(num_inputs, num_hidden, num_outputs)
optimizer = GradientDescent(lr=0.01)
```


```python
_ = visualize_classification(model, test_dataset.data, test_dataset.label)
```


    
![png](output_2_0.png)
    



```python
train_model(model, train_data_loader, optimizer)
eval_model(model, test_data_loader)
_ = visualize_classification(model, test_dataset.data, test_dataset.label)
```

    Model Accuracy: 100.00%



    
![png](output_3_1.png)
    

