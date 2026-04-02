"""
Configuration file for the project - contains all the parameters that can be easily changed

For the final results I used 384x768 resolution (can be changed to other resolutions),
 as it is the recommended size for Segformer B0 and B1, 
 which gained the best performance on the Cityscapes dataset, and are the models I used for the final results.
 They mostely where aimed for real live auto-driving applications, which is the main use case for the Cityscapes dataset, 
 and are the models I used for the final results.
 which are the models I used for the final results. 
 However, you can experiment with different sizes (e.g., 256x256 or 1024x1024) 
 to see how it affects the performance and inference time."""

HEIGHT = 384    # Segformer B0 and B1 work well with 384x768
WIDTH = 768     # Segformer B0 and B1 work well with 384x768

"""Baseline a UNet model was trained with 256x256 resolution,
Uncommand the following lines to use 256x256 resolution for the UNet model."""
# HEIGHT = 256
# WIDTH = 256

IMG_SIZE = (HEIGHT, WIDTH)
print(f"Image size: {IMG_SIZE}")