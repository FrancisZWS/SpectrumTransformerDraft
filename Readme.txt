This repository contains the code for trianing and testing Spectrum Transformer with DIY wideband PSD data (Data generation included).

When you use, please edit the file directories in these codes, especially the root directory 'Wireless_Transformer'.

For data generation, please use file 'DatasetGenerationUpdatedMat6Mod.ipynb' (file 'clean_PSD_6mod_Mat.pth' is needed, which stores the clean PSD of 6 modulation types, transformed from mat files in folder 'SpectrumLib'). 

I didn't put all my data in the repo. You can request from me or generate by yourself. We can discuss about data generation.

For training and testing, please use files starting with 'Multiple' (e.g., file 'Multiple_Vol_WrT_TestingWirelessTransformer_Updated.ipynb' means training and testing the performance of our Transformer on different data sizes. ) There are 6 files and their content are actually similar (train 100 epochs and plot/save accuracy, plot/save ROC after training). The key differences are their model architect and dataset selections. Results will be saved to different folders named by their structure and starting time. 

For the definition of Transformers, please refer to file 'wrt.py' ('vit.py' is the original Vision Transformer copied from https://github.com/lucidrains/vit-pytorch).

For definition of DeepSense and other DNNs, please check 'AlexNet1D.py'.