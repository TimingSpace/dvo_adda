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
# 20200317
1. modify align_corners = False to deal with a version warning
2. result on random training show that it is difficult to converge for full 6 dof motion

# 20200318
1. todo:  remap kitti to the same intrinsic parameter with random training data. to test the domain 
adaption performance.
2. add remap.py based on inverse_warp.py

# 20200319
1. todo :visualzie some inverse warped image to check and add to paper: 
    * available rgb and depth; 
    * available depth and random rgb
    * available rgb and random depth
    * random rgb and rangdom depth
# 20200320 

1. continue work of 0319
2. grid_corners has version conflicted, same code can't run both in 1.4 and 1.1 (solved by check the version then used
differient code )
3. control motion by motion.txt


# 20200321

# 20200322
1. never use -1 when reshape
2. add train on real image
