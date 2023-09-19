# Lesion masks
The grad-cam map can roughly locate the interest regions of CNN models. 

Generate grad-cam heatmaps of images, run:

	gradcam.py --model_name --dataset --cla_num --sample_num --seed

And the threshold controls the area of the mask. Thresholding and making a range of lesion masks, run:

	gen_mask.py --model_name --dataset --cla_num --up_val --down_val --step --seed

Select the appropriate threshold for your task. A higher threshold leads to a bigger space for optimizing the watermarks' location.

  
