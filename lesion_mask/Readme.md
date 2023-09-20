# Lesion masks
The grad-cam map can roughly locate the interest regions of CNN models. 

Generate grad-cam heatmaps of images, run:

	python gradcam.py --model_name --dataset --cla_num --sample_num --seed
 
Large sample numbers may cause an error 'out of the GPU memory'. The resized original image, cam, gb(guided-backpropagation), cam_gb, gray-scale masks will be saved in './MIAD-MARK/lesion_mask/results/[dataset]/[model_name]/[layer_name]'. 

And the threshold controls the area of the mask. Thresholding and making a range of lesion masks, run:

	python gen_mask.py --model_name --dataset --cla_num --down_val --up_val --step

Select the appropriate threshold for your task. A higher threshold leads to a bigger space for optimizing the watermarks' location. The different threshold masks will be saved in './MIAD-MARK/lesion_mask/results/[dataset]/[model_name]/[layer_name]+_threshold_mask'.

  
