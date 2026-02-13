# Spectrogram Pruning
## What is Spectrogram Pruning?
Spectrogram Pruning is the concept of removing irrelevant, redundant, or otherwise useless features. This is useful to clean up the input into a classifier/model. 

## Why in this context?
Our research problem is creating a classifier that distinguishes degraded from non-degraded reefs, given only their spectrograms. Due to the fact that we only have data from 3 different sites, our experiments often encode site-specific information rather than degradation-specific information. This repository contains my attempts to prune the spectrograms to remove site-specific information, and instead, focus on coral reef feature detections (ie snapping shrimp, fish, etc)

## What has been tried?
PCEN & Top-Hat. More information about either method is in their respective folder in their ReadMe. 
