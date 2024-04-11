# Conditioned Stimulus

## Example Usage:

### Environment

* Launch Anaconda Navigator
* Open Terminal with `conditioned_stim` environment activated
* Navigate to Analysis Pipeline Directory: `cd C:\Users\rober\Desktop\Analysis_Pipeline`

### add_video_script

```
python -m conditioned_stimulus.add_video_script C:/Users/rober/Desktop/Analysis_Pipeline/Monkey-Emotions/240110_aragorn_generalization_behave.pkl -o 240110_aragorn_video.pkl --video_folder C:\Users\rober\Desktop\Analysis_Pipeline\Monkey-Emotions\video\aragorn_240110 --max_load 10
```

python -m conditioned_stimulus.add_video_script "C:\Users\rober\Desktop\gandalf_20240112\240112_gandalf_VR_behave.pkl" -o 240112_gandalf_video.pkl --video_folder "C:\Users\rober\Desktop\gandalf_20240112\Gandalf_240112_Segmented" --max_load 10 --ignore_saved



python -m conditioned_stimulus.add_video_script "C:\Users\rober\Desktop\aragorn_20240306\240306_aragorn_airpuff_behave.pkl" -o 240306_aragorn_video.pkl --video_folder "C:\Users\rober\Desktop\Analysis_Pipeline\Monkey-Emotions\video\aragorn_240306" --ignore_saved

### decoding_comparison_script

```
python -m conditioned_stimulus.decoding_comparison_script C:\Users\rober\Desktop\Analysis_Pipeline\conditioned_stimulus\data\240110_aragorn_video.pkl --pre_pca 0.5python -m conditioned_stimulus.add_video_script C:/Users/rober/Desktop/Monkey-Emotions/240110_aragorn_generalization_behave.pkl -o 240110_aragorn_video.pkl --video_folder C:/Users/rober/Desktop/Monkey-Emotions/video/aragorn_240110
```

**Parameters**

> *--pre_pca_*: number (i.e. 100) or proportion (0.5) of PCA dimensions included in the analysis