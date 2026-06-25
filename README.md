# liverCancerSegmentator
Liver Cancer Segmentator: Metadata-Guided Confidence Scoring for Reliable Segmentation of Colorectal Liver Metastases in CT

“The associated model checkpoints will be released to the reviewers upon acceptance of the manuscript and can be shared upon request.”


# 1. clone repo
git clone https://github.com/yourname/liver-segmentation-ai
cd liver-segmentation-ai

# 2. install deps
pip install -r requirements.txt

# 3. download model from Hugging Face
python scripts/download_model.py

# 4. run inference
python src/inference.py \
--input test_input \
--output ./output_mask \
--model ./models/model/nnUNetTrainerV2__nnUNetPlansv2.1
