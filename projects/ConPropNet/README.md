#ConPropNet
Built off of D2's [TridentNet](https://github.com/facebookresearch/detectron2/tree/master/projects/TridentNet).


## General Hopes and Dreams
01. If COCO RPN could get most of the LVIS objects, it means that RPN can be
   sufficiently solved with a quick fix. Then, we embed every thing.
02. Else, do all the TO-DOs.

## TO-DOs 
00. ENSURE that as much of the detectron2 code is being used. Only do local
    imports where we need to modify default behavior. TridentNet did not
    reinvent the wheel. They only invented the behavior they needed to
    modify.
01. Do the metric that greedily matches GT labels to learnt RPN proposals.
02. Run forward pass and get proposals and sort by objectness.
    Insert the GT proposals at the top of the list.
    Run NMS.
    Pick top-N surviving proposals, treat them as anchors.
    Backpropagate.
    Repeat.
    Summary: Forward. Infer. Add GT with max_score. Run NMS. Target after NMS.
             Backprop. Repeat.

## Future TO-DOs
01. Figure out proposal and Deepak's paper.
02. Make RPNs class-aware.
03. Use the LVIS meta-data tags to backprop according to predicted classes.
04. Constrain based on hierarchy.
05. Check whatever Deepak wanted to do.
06. Check with Philipp what other constraints out of 4 were.




