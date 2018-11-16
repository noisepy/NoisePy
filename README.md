# NoisePy
Gathers all of our ambient noise modules in python


These libraries of codes are written in Python, format the data in ASDF, process ambient noise data in single and multi CPU processes. The workflow is (format ASDF):
- download the data
- compute all FFts w/o pre processing
- compute all noise correlations (auto and cross), saving all intermediate files
- stack with various techniques the cross
- (TBdone) process higher order correlations, pre and post stacks.
- perform various estimates of dv/v (stretching/MWCS/DTW)


Contributing authors are:
- Tim Clements, Chengxin Jiang, Marine Denolle
