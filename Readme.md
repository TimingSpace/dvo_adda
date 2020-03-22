# Deep Visual Odometry With Adversarial Domain Adaption

this code is based on `https://github.com/corenel/pytorch-adda` trying to investigate the domain adaption for learning
based visual odometry problem.

The assumption of this world is that: 

1. we have labelled data for domain A(from such as public dataset, or synthetic data) 
2. we have unlabelled data for domain B (the domain is where you want to deploy your algorithm)
3. there are some shift or difference between domain A and domain B ( weather condition, scenario difference)
4. the camare intrinsic parameter for both domain is available

## Requirement
* Python 
    * Pytorch numpy scipy pandas scikit-image
* GPU

## Network Structure
* VO Feature Extraction
* VO regression
* Discriminator

## Image Processing
1. intrinsic remapping

## Random Dataset
We proposed to train a deep visual odometry model with a pure random dataset created by following procedure:
```
image_1 = np.random.random((1,self.camera_parameter[1],self.camera_parameter[0]))
```
```
depth   = np.random.random((1,self.camera_parameter[1],self.camera_parameter[0]))
```
```
motion  = np.random.random((6))   
image_2 = remap(image_1,depth,motion)
```

## Result
