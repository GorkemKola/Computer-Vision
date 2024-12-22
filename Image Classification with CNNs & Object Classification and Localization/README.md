Please specify the location (e.g. src/ or src/mypackage) 
where your implementation should be runned at:
/////////// Type Below: LOCATION OF CODE //////////
src
///////////////////////////////////////////////////
Please specify the command that is needed 
to run your implementation:
(e.g. python3 main.py)

///////////// Type Below: COMPILATION //////////////

It can be run just doing

python -m src.main
--data_dir: LOCATION_OF_IMAGES
--train_paths_file: LOCATION_OF_TRAIN_TXT
--test_paths_file_: LOCATION_OF_TEST_TXT
--part1: runs part1
--part2: runs part2
--part3: runs part3
--clean_data: preprocess txt files
--learning_rate: learning rates will be used
--batch_size: batch sizes will be used
--dropout_rate: dropout rates will be used
--verbose: enable verbose

But as arguments can be given, an example is given below

python -m src.main --data_dir LOCATION_OF_IMAGES --train_paths_file LOCATION_OF_TRAIN_TXT --test_paths_file LOCATION_OF_TEST_TXT --part1 --part2 --part3 --clean_data --learning_rate 0.01 0.001 0.0001 --batch_size 16 128 --dropout_rate 0.2 0.4 0.6 0.8 --verbose

Also there is a plot file in scripts, and it can be run using this command:
python -m src.scripts.plot
--history_dir: LOCATION_OF_HISTORIES
--save_to: WHERE_TO_SAVE_PLOTS(png is recommended.)
--model_name: custom or vgg
--learning_rate: a learning rate will be plotted
--batch_size: a batch size will be plotted
--dropout_rates: if true just uses dropout added ones
--all: if true plots all

it can be run like:

python -mm src.scripts.plot --history_dir LOCATION_OF_HISTORIES --save_to WHERE_TO_SAVE_PLOTS --model_name custom --learning_rate0.001


////////////////////////////////////////////////////

Please specify the location (e.g. src/ or src/mypackage)
where the dataset folder should be placed at:

///////// Type Below: LOCATION OF DATASET //////////

It set src/data/indoorCVPR_09/Images as default but if it is wanted to be changed you can run python using
python -m src.main --data_dir LOCATION_OF_IMAGES --train_paths_file LOCATION_OF_TRAIN_TXT --test_paths_file LOCATION_OF_TEST_TXT

////////////////////////////////////////////////////

Please specify the Python 3 version, libraries and
their versions which are mandatory to run your code
//////// Type Below: LIBRARIES AND VERSIONS ////////
I use Python 3.10.13, 
and necessary libraries can be downloaded using
pip install -r src/requirements.txt

BUT AS WINDOWS DOES NOT SUPPORT Tensorflow 2.15, Ubuntu should be used. If you want to use Windows please first read the tensorflow installation documentation.
////////////////////////////////////////////////////

Please specify the details and organization of your
implementation and extra information if you have any
//// Type Below: DETAILS AND EXTRA INFORMATION /////
- b2200765032.zip
    - README.md
    - Report.pdf
    - src/
        - data/(Location of data can be changed)
            - TestImages.txt
            - TrainImages.txt
            - indoorCVPR_09/
                - Images/
                    - *
            - indoorCVPR_09annotations/
                - Annotations/
                    - *
        - main.py
        - dataset.py
        - config.yml
        - requirements.txt
        - models.py
        - part1.py
        - part2.py
        - part3.py
        - plot.py
        - test.py
        - train.py

* Part2 needs the Part1 ran before running. So, please first run part1 or run them together.
* Could not put the output folder because of the submit system and as there are so many of them putting them on the assignment folder would be messy, so they are uploaded on the drive here is the link:
https://drive.google.com/drive/folders/1CfkTvsdJNBvuEbyRkDIqNmloAVoaQRpb?usp=sharing 
////////////////////////////////////////////////////
