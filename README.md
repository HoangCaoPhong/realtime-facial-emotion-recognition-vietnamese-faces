# VN-FER-Realtime
Vietnamese Facial Emotion Recognition with Domain Adaptation

## Motivation
Most facial emotion recognition models are trained on Western-centric datasets
(e.g. FER2013, collected in 2013). When applied to Vietnamese faces, performance
degrades due to domain shift in facial structure and expression patterns.

This project investigates domain adaptation via fine-tuning on Vietnamese facial data.

## Contributions
- Built a real-time FER system using CNN + OpenCV
- Identified dataset bias in FER2013 when applied to Vietnamese faces
- Designed fine-tuning pipeline with Vietnamese facial data
- Improved emotion recognition accuracy on VN test set
- Provided reproducible training & evaluation scripts

## System Overview
Webcam → Face Detection → Preprocessing → CNN → Emotion Prediction

## Dataset
- Base dataset: FER2013 (pretrained model)
- Vietnamese subset: self-collected + manually verified labels
- 7 emotion classes

## Methodology
- Transfer learning from pretrained CNN
- Freeze backbone, train classifier head
- Partial unfreeze with low learning rate
- Data augmentation

## Evaluation
- Accuracy
- Confusion matrix
- Comparison: pretrained vs fine-tuned

## Future Work
- Replace Haar Cascade with MTCNN
- Larger Vietnamese dataset
- Cross-cultural emotion analysis
