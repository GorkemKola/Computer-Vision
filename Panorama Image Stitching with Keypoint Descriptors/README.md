Please specify the location (e.g. src/ or src/mypackage) 
where your implementation should be runned at:
/////////// Type Below: LOCATION OF CODE //////////

src/

///////////////////////////////////////////////////
Please specify the command that is needed 
to run your implementation:
(e.g. python3 main.py)
///////////// Type Below: COMPILATION //////////////

To run the implementation, use these commands: 
python3 src/main.py --dataset "Dataset Root Directory" --output "Output Directory" --verbose
--dataset: to demonstrate dataset location.
--output: to demonstrate output location
--verbose: to print informations while running.
////////////////////////////////////////////////////

Please specify the location (e.g. src/ or src/mypackage)
where the dataset folder should be placed at:
///////// Type Below: LOCATION OF DATASET //////////
The Root Directory have to be
src/"Dataset Root Directory"/*

Dataset should be located on a root directory. The name of dataset do not have to be specified. 

////////////////////////////////////////////////////

Please specify the Python 3 version, libraries and
their versions which are mandatory to run your code
//////// Type Below: LIBRARIES AND VERSIONS ////////
Python 3.6.15
opencv-contrib-python==3.4.2.16 is necessary, as opencv 4 do not have SURF. 
To install all libraries, run: pip install -r src/requirements.txt
////////////////////////////////////////////////////

Please specify the details and organization of your
implementation and extra information if you have any
//// Type Below: DETAILS AND EXTRA INFORMATION /////
My implementation is in the src directory.

I use Python 3.6.15, and the required libraries are specified in src/requirements.txt.

Implementation Organization:
- b2200765032.zip
  - README.md
  - Report.pdf
  - src/
      - features.py
      - homography.py
      - main.py
      - stitch.py
      - util.py
      - warp.py
      - requirements.txt
      - dataset/* (Sub datasets such as fishbowl and carmel.) (If you run main.py outside the src, dataset should be located there.)

////////////////////////////////////////////////////
