    class_wise_metrics         = Reads in lvis_instance_results.json and compares to LVIS GT to
                                 generate class-wise APs. Also stores, the 4D precisions tensor.
    convert_lvisval_to_cocoval = Convert files of the form COCO_val2014_XXXXXXXXXXXX.jpg to XXXXXXXXXXXX.jpg.
    infer_all                  = Give chkpt dir and exp config, run inference on all available chkpts.
    manipulating_annotations   = Generate CNL: Subset of COCO train consisting of same images as LVIS_v0.5.
    switching_rpn_n_class      = Create 2 checkpoints where converged COCO and LVIS models have their {backbone+rpn}
                                 and {roi_heads} exchanged.
    vis_utils                  = Migrated from mmlab.
    upper_bound_map            = take box proposals. take GT JSONs. greedily map one proposal to one proposal.
                                 compute upper bound on MAP where bottleneck will be localization.
