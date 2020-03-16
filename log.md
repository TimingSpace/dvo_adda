# 20200313
1. test training, loss go from 0.x to 0.0x

# 20200314
1. check random image
2. add motion control

# 20200315
1. add log
2. add option
3. train '111111' for 1000 epoch(10 data each epoch) loss is 0,00x (unfortunately deleted)
4. train 001000 001010 111010 111111

# 20200316
1. add save model
2. add test
3. found the loss keep same for each epoch when testing, random is not random 
4. solve random problem: `worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid))`
5. add gpu train
6. len(dataloader) = len(dataset)/batch_size
# 20200316
