# Semantic Face Segmentation using ML techniques

The work is primarily based on Khan, K., Mauro, M., & Leonardi, R. (2015). Multi-class semantic segmentation of faces. 2015 IEEE International Conference on Image Processing (ICIP), 827-831.

The difference is the addition of facial landmarks as a feature.

## Requirements

This project requires `torch`, `scikit-learn`, `scikit-image`, `numpy`, `tqdm`, `opencv-python`.

## Dataset

The dataset used is LaPa-Dataset for face parsing. The link for download is available at its [homepage](https://github.com/jd-opensource/lapa-dataset).

Place the dataset in a directory named `LaPa` in the root of the project. The structure of the project will then look like this:
```
LaPa/
    test/
        images/
        labels/
        landmarks/
    train/
        ...
    val/
        ...
src/
    data/
        ...
    main.py
    rf.py
    mlp.py
```

## Procedure

Generate per-pixel features from `n` images by first adjusting the `main.py` file according to your needs and computational capabilites. Beware that for `n = 5`, i.e. for 5 images, the process of feature extraction takes about a minute with 64 cores. Set `num_workers` to a number of CPUs on your machine to make this process as fast as possible.

For our experiments, we've generated a training set of samples from 5 images and a test set from 3 images. These images take up upwards of 45GB in RAM when uncompressed. If you want to save 1 image per training and test set, make sure you **change the filename** set in `main.py`.

Finally, run either of `mlp.py` or `rf.py` (with minor adjustments to filenames, if any) to see the performance. In `rf.py`, one should also adjust the number of workers according to their machine specification.

### Attribution

Khan, K., Mauro, M., & Leonardi, R. (2015). Multi-class semantic segmentation of faces. 2015 IEEE International Conference on Image Processing (ICIP), 827-831.

```
@article{Khan2015MulticlassSS,
  title={Multi-class semantic segmentation of faces},
  author={Khalil Khan and Massimo Mauro and Riccardo Leonardi},
  journal={2015 IEEE International Conference on Image Processing (ICIP)},
  year={2015},
  pages={827-831},
  url={https://api.semanticscholar.org/CorpusID:1408465}
}
```