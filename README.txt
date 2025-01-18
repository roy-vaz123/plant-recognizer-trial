1) orgenize  houseplants_data DB: divide data to trainig and validation
	Training Data (train):
		Purpose: The model learns from this dataset by adjusting its weights.
		Process:
		The model takes input images, processes them through multiple layers, and predicts output (e.g., plant species).
		The loss function calculates the error between the prediction and the actual label.
		The optimizer updates the weights based on the loss to improve accuracy.
		Size: Typically 70-80% of the total dataset.
	Validation Data (val):
		Purpose: Used to evaluate model performance on unseen data during training.
		Why? The model should generalize well to new data. If it only performs well on train but poorly on val, it is likely overfitting.
		No Weight Updates: The model does not learn from this dataset—weights are not updated.
		Size: Usually 10-20% of the dataset.

2)run train_model.py only once (in theory):
	will use houseplants_data (need to be filled according to format with classified plant pictures)
	to fine tune a pre trained image classify deep learning model (using pytorch model: resnet50, maybe change for a smaller one)
  training performance:
	Dataset Size:	A small dataset (~100-1000 images) is fine for most laptops. Larger datasets (e.g., millions of images) require more power.
	Model Size:	Simple models (e.g., a small CNN) can run on a CPU. Large models (e.g., ResNet50, Transformer models) benefit from a GPU.
	Number of Epochs: 	More training epochs = longer training time.
	Batch Size:	Large batch sizes need more RAM (GPU memory).

	ResNet50 with 224x224 RGB Images memory usage:
	Batch Size	Memory per Batch (FP32)
	1		~25 MB
	16		~400 MB
	32		~800 MB
	64		~1.6 GB
	128		~3.2 GB
	
	fine-tune ResNet50 on a CPU using 200-500 images (just a random number that seems logical :))
	Estimated Training Time on CPU:
	Batch Size	Approx. Time per Epoch
	32		15-30 minutes
	16		20-45 minutes
	8		30-60 minutes



3)run classify image:
	prints to console the image and class (plant type) of every image
