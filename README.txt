Source code used in the work:
"High-accuracy prostate cancer pathology using deep learning"
by Tolkach Y. et al.

### 1) System requirements ###
*Operating system
Ubuntu 16.04 / 18.04

*Python 3.6.5

*Libraries / Dependencies
tensorflow 1.x (GPU version); also functionable with 2.0; also tested on CPU version
keras 2.2.5
staintools 2.1.2
openslide 3.4.1
PIL 7.0.0
Additional packages: statistics, numpy, opencv, os (last versions)




### 2) Hardware requirements ###
The code was implemented on the PC with Nvidia GPU card.
No non-standard hardware required.




### 3) Installation guide ###
The source code is not intended for "installation" per se.
Typical install time: not applicable.
The source code could be executed in the command line or in GUI software
(Spyder, jupyter notebook)




### 4) Models ###
Trained models are not provided due to their inherent value.




### 5) Source code modules (correspond to folders) ###

List of modules
1 Training
2 validation Tumor vs Benign
3 validation Gleason score
4 WSI pipeline
5 Gleason score minimal tumor size


                        SOURCE CODE MODULES IN DETAILS

# 1 Training #
*Training of the models (model type 1: tumor vs benign, model type 2: gleason patterns)
*For detailed description of the training approach see Manuscript.
*Source code for training is similar for both model types
(3 classes, s. Details in the Manuscript).

Comments:
-Only Version for NASNetLarge network provided at magnification of approx. 23x
-Representative images from training dataset included (stain normalized)
-Graph architecture of the network is provided (NASNetLarge with shapes.png).





# 2 Validation Tumor vs Benign #
*Validation of the model type 1 using two validation datasets
(Discrimination abilities between tumor and benign tissue).
*Validation datasets (patches of size 600px at 40x) are generated through
tiling of tumor and benign regions from pathologist pre-annotated whole slide images.

*Functionality:
- Test can include one or more trained models
- Takes images (patches) from source folder and generates predictions
using model 1 (discrimination tumor vs benign tissue).
- Patches with predictions in gray zone (see Methods)
are being additionaly analysed through C8 approach (see Methods; shortly:
analysis not only of native patch but also of its derivates: flips and rotations).
- Generated results are saved as tab-delimited file
- Script can be adapted to any thresholds to trigger C8 testing (including all
images without any thresholds)





#3 Validation Gleason Score
*Validation of the model type 2 using three validation datasets (s. Methods).
*Validation datasets for this purpose are different to Validation datasets for
setting Tumor vs Benign.
*Validation datasets contain large tumor images up to 4800 px and more
(no more patches; saved under magnification 40x).


*Functionality:
- Large images are being splitted into patches 600x600 px, which are further
reduced to model working patch size.
- Every patch is analysed by model type 2 (predictions for Gleason patterns 3, 4 and 5)
- Final grading of the whole tumor area according to developed algorithm
(output: percentages of gleason patterns; further will be transformed to Gleason scores)
- Saving the results into a tab-delimited file (would be further merged
with clinical database for agreement analysis and survival-based analysis)





#4 WSI pipeline
A pipeline for processing of whole-slide images.
NB! Trained models are not provided due to their inherent value.

$ Functionality:
- Processes the whole slide image (tiling, background/tissue detection, classification,
creation of maps and overlays)
- Final outputs of the pipeline in the provided version are:
  *mathematical maps with predictions and classification results
  *binary maps with classification results: background/tissue AND tumor vs benign;
  Gleason grading maps
  *overlaid images: heatmaps overlaid on the reduced whole-slide images

$ Processes during implementation:
- cutting WSI parts with tissue into patches 600x600 px.
- reduction of patch size to model patch size (350x350 px).
- background / tissue detection (based on the presence of the staining patterns
typical for cell nuclei)

*if patch = non-background
- brightness and stain normalization of the patch (Macenko method)
- test of patch via model type 1 with generation of probabilities of classes
(tumor or benign)
- creation of mathematical map based on probabilities for tumor and benign tissue
using threshold 0.5
- implementing additional strategies to reduce false positive results
(Strategy C8 and single patch environment analysis; see Materials and methods
for description)
- generation of the final binary map of the tumor vs benign tissue
- genetation of the image overlay: binary map on the reduced version of the
whole slide image.

- analysis of the tumor classified patches with the model type 2 for Gleason grading
(details of the algorithm for gleason Grading see Materials and Methods)
- generation of the Gleason grading map.
- genetation of the overlay: binary Gleason grading map on the reduced version of the whole slide image.
- save binary maps, heatmaps and overlaid images

$ Code components and description.
*Folders
images                         Contains chunk images for binary (heatmap) maps
                               Contains standard image for stain normalization

*Files:
MAIN.py                        Main script to execute pipeline
                               (Magnification of analysis defined through
                               model patch size)
wsi_c8_functions.py            Implementation of C8 strategy (s. Methods)
                               - generation of patch flip and rotation derivates
                               - final prediction for patch based on analysis
                               of derivates
wsi_calculate_square.py        Calculation of tumor square in the slide
wsi_detectors.py               Tissue / background detection functions
wsi_gleason.py                 Single patch gleason grade prediction
                               Algorithm for Gleason scoring from pattern predictions
                               Binary Gleason score map generation
wsi_heatmaps.py                Load images for binary map/heatmap for
                               different classes
wsi_maps.py                    Make mathematical binary map from predictions,
                               Make binary heatmap as image
                               Make overlaid image
wsi_process.py                 Main script to process whole slide image
                               Separate implemetation for C1 (native) and
                               C8 strategies (s. Methods)
                               Processes:
                               -tiling
                               -background / tissue detection
                               -brightness/stain normalization
                               -predictions
                               -make mathematical map with predictions (numpy array)

wsi_single_env.py              Implementation of additional "environment" analysis
                               of single positive patches (s. Methods)
wsi_slide_info.py              Retrieve slide info from metadata
wsi_stain_norm.py              Initialization of Brightness and Stain normalizers

*Comments:
- WSIs from different scanner vendors may require small modifications of
wsi_slide_info.py to cope with different metadata saving methods






#5 Gleason score minimal tumor size

*Analysis of the minimal tumor size necessary for reliable Gleason Grading.
*Refers to data presented in the Figure 6
*See Methods for description of principle.
Shortly:
-Gleason grading of subsampled ROIs from large tumor images
-Progressively increasing size of ROIs (in patches)
-Bootstrapping for every ROI size in patches (20 rounds).
-Comparison of the grading of subsampled ROIs and the
grading of the whole image.
-Main Aim: at which smallest tumor size grading becomes reliable, representative
of the whole tumor area.

Principle (See also Methods):
- Dataset of large tumor images (Examples provided).
- Random extraction of ROIs with a size of 1, 2, 3, n ... patches from large tumor images.
  *Random generation of coordinates.
- Bootstrapping with 20 iterations pro ROI size (number of subsampled patches).
- Control for non-intersecting patch extraction.
- Control for maximal image size to exclude some images from analysis with large ROI sizes.
- Gleason grading according to algorithm described in Methods and implemented in the
WSI pipeline.
- Saving grading results for every bootstrapping iteration for further comparison to grading
results of the whole tumor area in the image.
