	epi_1 = This is basically Xingyi's code adapted for LVIS.
        	We only work with GT annotations. This script does not parse predictions.
	epi_2 = In epi_1: Add a canvas around each image to allow ffmpeg to do it's thing.
	epi_3 = Works only for bbox outputs of lvis_test.py
	epi_4 = Merge epi_2 and epi_3. Produce a single video with GT and preds on left and
		right, respectively.

