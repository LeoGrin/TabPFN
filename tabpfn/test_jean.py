import torch
import torchvision
import idr_torch # IDRIS package available in all PyTorch modules
from tabpfn.priors.trees import dataset
 
 

 
# define distributed sampler

 
# define DataLoader 
batch_size = 128                       # adjust batch size according to GPU type (16GB or 32GB in memory)
drop_last = True                       # set to False if it represents important information loss
num_workers = 4                        # adjust number of CPU workers per process
persistent_workers = True              # set to False if CPU RAM must be released
pin_memory = True                      # optimize CPU to GPU transfers
non_blocking = True                    # activate asynchronism to speed up CPU/GPU transfers
prefetch_factor = 2                    # adjust number of batches to preload
 
dataloader = torch.utils.data.DataLoader(dataset,
                                         #sampler=data_sampler,
                                         batch_size=batch_size, 
                                         drop_last=drop_last,
                                         num_workers=num_workers, 
                                         persistent_workers=persistent_workers,
                                         pin_memory=pin_memory,
                                         prefetch_factor=prefetch_factor
                                        )
 
# loop over batches
for i, (images, labels) in enumerate(dataloader):
    images = images.to(gpu, non_blocking=non_blocking)
    labels = labels.to(gpu, non_blocking=non_blocking) 
