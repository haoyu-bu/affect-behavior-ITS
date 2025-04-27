# Multimodal Affect Behavior Prediction in an Intelligent Tutoring System

Source code for the following papers:

* Hao Yu, Danielle A. Allessio, William Rebelsky, Tom Murray, John J. Magee, Ivon Arroyo, Beverly P. Woolf, Sarah Adel Bargal, Margrit Betke. Affect Behavior Prediction: Using Transformers and Timing Information to Make Early Predictions of Student Exercise Outcome. In 25th International Conference on Artificial Intelligence in Education (AIED), 2024. [[link]](https://link.springer.com/chapter/10.1007/978-3-031-64299-9_14)

* Hao Yu, Danielle A. Allessio, Will Lee, William Rebelsky, Frank Sylvia, Tom Murray, John J. Magee, Ivon Arroyo, Beverly P. Woolf, Sarah Adel Bargal, and Margrit Betke. COVES: A Cognitive-Affective Deep Model that Personalizes Math Problem Difficulty in Real Time and Improves Student Engagement with an Online Tutor. In Proceedings of the 31st ACM International Conference on Multimedia (ACM MM), 2023. [[link]](https://dl.acm.org/doi/10.1145/3581783.3613965)

## Installation

Install the packages:
```bash
conda create -n coves python=3.10
conda activate coves
pip install -r requirements.txt
```

## Data Pre-processing

* Extract video frames (downsample if needed, we used 3fps in this work) and store all frames in ```data/raw_images```.
* Detect and crop faces in video frames using Dlib. Download [mmod_human_face_detector.dat](https://dlib.net/files/mmod_human_face_detector.dat.bz2) in ```models```.
```bash
python src/script/extract_faces.py
```
* Extract facial features using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). Generate a csv file for each video in ```data/openface/```.
* Process facial features.
```bash
python src/script/process_facial_features.py
```
* Extract affect features using an affect network pretrained on [AffectNet](https://arxiv.org/abs/1708.03985). The pretrained weights can be downloaded from [here](https://drive.google.com/file/d/1jKTvN1AXZFAR6O9-xL7jVRjiuA2xrI9K/view?usp=sharing).
```bash
python src/script/extract_affect_features.py /PATH/FACES/
```
* Generate training and testing split with feature paths. We use 5-fold cross validation by generating 5 training and testing csv files. Each file contains
  * videoID
  * Multiple meta information fields
  * af_features_path
  * openface_features_path
  * effort or next_effort (target labels)

## Train

To train COVES, run the following:
```bash
python train_coves.py
```

To train the early prediction model, run the following:
```bash
python train_early.py SECONDS NCLASS ATTN
```
SECONDS specifies the data used for the prediction (5, 10, 15, 20).

NCLASS specifies the number of classes (2 or 7).

ATTN specifies whether to use attention-based fusion method (True or False).

## Citation

If you find it useful for your research and applications, please cite related papers using this BibTeX:

```bibtex
@inproceedings{yu2024affect,
  title={Affect Behavior Prediction: Using Transformers and Timing Information to Make Early Predictions of Student Exercise Outcome},
  author={Yu, Hao and Allessio, Danielle A and Rebelsky, William and Murray, Tom and Magee, John J and Arroyo, Ivon and Woolf, Beverly P and Bargal, Sarah Adel and Betke, Margrit},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={194--208},
  year={2024},
  organization={Springer}
}

@inproceedings{yu2023coves,
  title={COVES: A Cognitive-Affective Deep Model that Personalizes Math Problem Difficulty in Real Time and Improves Student Engagement with an Online Tutor},
  author={Yu, Hao and Allessio, Danielle A and Lee, Will and Rebelsky, William and Sylvia, Frank and Murray, Tom and Magee, John J and Arroyo, Ivon and Woolf, Beverly P and Bargal, Sarah Adel and Betke, Margrit},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={6152--6160},
  year={2023}
}
```
