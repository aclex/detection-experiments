import torch

def get_mean_std(dataset):
	'''Compute the mean and std value of dataset.'''
	dataloader = torch.utils.data.DataLoader(dataset)

	mean = torch.zeros(3)
	std = torch.zeros(3)

	print('Computing mean and std...')
	for d in dataloader:
		inputs = d["image"]
		for i in range(3):
			mean[i] += inputs[:,i,:,:].mean()
			std[i] += inputs[:,i,:,:].std()

	mean.div_(len(dataset))
	std.div_(len(dataset))

	return mean, std
