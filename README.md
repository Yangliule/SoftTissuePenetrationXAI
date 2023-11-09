# LiuleYangSemesterProject

## NOTE: This repo removed original data (inference_data, training_data), since they are private properties. Only retained extracted data and code implementation here.

## Structure

1. extracted_data: stores the extracted features files after the data_pipeline.
2. inference_data: stores recording files for inference. IMPORTANT: the recording contained in the folder now is a dummy one. You need to replace it to make meaningful inference.
3. training_data: stores recording files for training.
4. src: code base for python file. 
    - data_pipeline.py: contains feature extraction pipeline
    - predict.py: used for performing inference using the second GAM model in the analysis_report notebook. You can change the model to any of the model in the analysis report to perform inference.
    - counterfactual.py: perform conterfactual analysis using features selected according to the analysis_report notebook. The model is ridge regression since GAM models are not supported by counterfactual analysis package.
5. Analysis_notebook: the 'scratch paper'
6. analysis_report: the notebook for behavior analysis. 
7. data_pipeline: the notebook for feature extraction data pipeline.

## How to run notebooks:
1. create virtual environment with conda env create -f environment.yml
2. activate the virtual enviroment with conda activate behavior_analysis
3. Then you can run all notebooks cell by cell using the envivronment you just created. First run the data_pipeline notebook, and then run the analysis_report notebook. In the notebook, you will be able to see/edit all the engineering process.

## Make inference and generate reports for new recordings using the python files
1. Make sure to create and activate the virtual environment from the environment.yml file.
2. Make sure to put the recording file (json) and event_protocal for that recording (txt) into inference_data folder. Also, make sure each recording has three drills in total
3. To perform prediction:
    - run from root directory: python src/data_pipeline.py --training_or_inference training --saving_directory extracted_data. This step is to run the feature extraction pipeline for training data. Please make sure to rerun this when you change the data file in training_data folder
    - run from root directory: python src/data_pipeline.py --training_or_inference inference --saving_directory extracted_data
    - run python src/predict.py from root directory. The result will be shown in the command line prompt.
4. To run counterfactual analysis:
    - run from root directory: python src/data_pipeline.py --training_or_inference training --saving_directory extracted_data (if you haven't done this when performing prediction)
    - run from root directory: python src/data_pipeline.py --training_or_inference inference --saving_directory extracted_data (if you haven't done this when performing prediction)
    - run python src/counterfactual.py --lower_bound [lower_bound_of_penetration] --upper_bound [upper_bound_of_penetration]. Note that in the comman, the lower bound and upper bound is the bound you WANT the penetration to fall in. Please also note that if the process is taking too long, you might want to cancel the running and choose a different range. The default range is [5.0, 15.0].
5. IMPORTANT note: the python file only generates prediction and counterfactual analysis results. To have a full understanding of the modeling process, please run the analysis_report notebook, which demonstrates the full thinking process.
