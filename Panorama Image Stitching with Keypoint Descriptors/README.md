/////////// WHAT IS THIS PROJECT //////////

This project is trying to solve the image stitching problem. The code finds keypoints (Blobs) then apply RANSAC to find appropriate homography matrix and inliers, then stitch images recursively.

/////////// LOCATION OF CODE //////////

src/

///////////// COMPILATION //////////////

To run the implementation, use these commands: 
python3 src/main.py --dataset "Dataset Root Directory" --output "Output Directory" --verbose
--dataset: to demonstrate dataset location.
--output: to demonstrate output location
--verbose: to print informations while running.

///////// LOCATION OF DATASET //////////

The Root Directory have to be
src/"Dataset Root Directory"/*

Dataset should be located on a root directory. The name of dataset do not have to be specified. 

//////// LIBRARIES AND VERSIONS ////////

Python 3.6.15
opencv-contrib-python==3.4.2.16 is necessary, as opencv 4 do not have SURF. 
To install all libraries, run: pip install -r src/requirements.txt

//// DETAILS AND EXTRA INFORMATION /////

The implementation is in the src directory.

I use Python 3.6.15, and the required libraries are specified in src/requirements.txt.

Implementation Organization:
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
