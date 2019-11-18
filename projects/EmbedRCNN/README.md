#Embed-RCNN

Built off of D2's [TridentNet](https://github.com/facebookresearch/detectron2/tree/master/projects/TridentNet).


## General Idea
1. Adapt the Generalized RCNN (Fast/ Faster/ Mask) to the embedding learning problem.
2. Change as few things as possible.
3. Modify the following:
    Data sampler must not only sample rare classes with RFS, but also sample tuples of the same class.
    Classification ROI head now learns embeddings and optimizes TripletLoss, instead of SoftmaxLoss.
4. Ensure that we do justice to the baselines as well. If new sampler breaks baseline, we should be aware of it.


## TO-DOs
1. Expose embeddings from ROI-Head and get basic embedding learning pipeline going on a regular RFS sampler.
   This would result in the contrastive loss, not in the triplet loss.
2. Eventually, start sampling for embeddings, which allows triplet learning.



