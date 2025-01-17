1)run train_model.py only once (in theory):
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

2)run classify image:
	prints to console the image and class (plant type) of every image
