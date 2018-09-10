
Example to run prediction:
saved_model_cli run --dir trained_models/trashNet_v1/1536545312 --tag_set serve --signature_def serving_default --input_exprs='image_filename="dataset/plastic/plastic10.jpg"'