# Conditioned Stimulus

## Example Usage:

### add_video_script

```
python -m conditioned_stimulus.add_video_script C:/Users/rober/Desktop/Monkey-Emotions/240125_aragorn_generalization_behave.pkl -o 240125_aragorn_video.pkl --video_folder C:/Users/rober/Desktop/Monkey-Emotions/video/aragorn_240125
```

### decoding_comparison_script

```
python -m conditioned_stimulus.decoding_comparison_script C:\Users\rober\Desktop\Analysis_Pipeline\conditioned_stimulus\data\240110_aragorn_video.pkl --pre_pca 0.5python -m conditioned_stimulus.add_video_script C:/Users/rober/Desktop/Monkey-Emotions/240110_aragorn_generalization_behave.pkl -o 240110_aragorn_video.pkl --video_folder C:/Users/rober/Desktop/Monkey-Emotions/video/aragorn_240110
```

**Parameters**

> *--pre_pca_*: number (i.e. 100) or proportion (0.5) of PCA dimensions included in the analysis