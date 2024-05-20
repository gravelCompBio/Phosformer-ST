<!-- This github was Made by Nathan Gravel --> 

# Phosformer-ST  <img src="https://github.com/gravelCompBio/Phosformer-ST/assets/75225868/f375e377-b639-4b8c-9792-6d8e5e9e6c39" width="60"> 

  

## Introduction   

  

   


  

   

  

This repository contains the code to run Phosformer-ST locally described in the manuscript "Phosformer-ST: explainable machine learning uncovers the kinase-substrate interaction landscape". This readme also provides instructions on all dependencies and packages required to run Phosformer-ST in a local environment. 
</br> 

   

## Quick overview of the dependencies 

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

  

![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) 
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) 

  

   

</br> 

  

  

## Included in this repository are the following:   

  

   

  

- `phos-ST_Example_Code.ipynb`: ipynb file with example code to run Phosformer-ST 

  

    - `modeling_esm.py`: Python file that has the architecture of Phosformer-ST 
    
      
    
    - `configuration_esm.py`: Python file that has configuration/parameters of Phosformer-ST  
    
      
    
    - `tokenization_esm.py`: Python file that contains code for the tokenizer  

  

  

- `multitask_MHA_esm2_t30_150M_UR50D_neg_ratio_8+8_shift_30_mask_0.2_2023-03-25_90.txt`: this txt file contains a link to the training weights held on the hugging face or zenodo repository 

    - See section below (Downloading this repository) to be shown how to download this folder and where to put it
  

- `phosST.yml`: This file is used to help create an environment for Phosformer-ST to work 

   

- `README.md`: 

  

- `LICENSE`: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License 

  

  

    

  

</br> 

  

</br> 

  

    

  

## Installing dependencies with version info    

  

  

### From conda:    

  

![python=3.9.16](https://img.shields.io/badge/Python-3.9.16-green)  

  

![jupyterlab=4.0.0](https://img.shields.io/badge/jupyterlab-4.0.0-blue)  

  

Python == 3.9.16  

  

   

  

### From pip:  

  

   

  

![numpy=1.24.3](https://img.shields.io/badge/numpy-1.24.3-blue)  

  

![pandas=2.0.2](https://img.shields.io/badge/pandas-2.0.2-blue)  

  

![matplotlib=3.7.1](https://img.shields.io/badge/matplotlib-3.7.1-blue)  

  

![scikit-learn=1.2.2](https://img.shields.io/badge/scikitlearn-1.2.2-blue)  

  

![tqdm=4.65.0](https://img.shields.io/badge/tqdm-4.64.1-blue) 

  

![fair-esm=2.0.0](https://img.shields.io/pypi/v/fair-esm?label=fair-esm)   

  

![transformers=4.31.0](https://img.shields.io/badge/transformers-4.31.0-blue)  

  

![torch=2.0.1](https://img.shields.io/badge/torch-2.0.1-blue)      

  

### For torch/PyTorch 

  

Make sure you go to this website https://pytorch.org/get-started/locally/ 

  

Follow along with its recommendation  

  

Installing torch can be the most complex part  

  
  
  

  

</br> 

  

</br> 

  

   

  

## Downloading this repository   

  

```   
gh repo clone gravelCompBio/Phosformer-ST 
```   

  

```   
cd Phosformer-ST 
``` 

### The following step demonstrates users how to download the training weights 


  -other repositories were used because the folder's memory size is larger than the allowed space on github 

  

  

</br> 

  

### Main option) Hugging Face  

  

Then download the link found in `multitask_MHA_esm2_t30_150M_UR50D_neg_ratio_8+8_shift_30_mask_0.2_2023-03-25_90.txt` or can be found at this link https://huggingface.co/gravelcompbio/Phosformer-ST_trainging_weights/tree/main 

  

The download link should take to a page that should look like this 

  

  

![Screenshot from 2023-07-24 13-49-54](https://github.com/gravelCompBio/Phosformer-ST/assets/75225868/bd2ebb5e-6174-4695-9cd3-730b835a8664) 

  

  

  

Click the download box highlighted in picture above 

  

  

  

</br> 

  

### Alternative option) Zenodo  

  

  

Then download the link found in `multitask_MHA_esm2_t30_150M_UR50D_neg_ratio_8+8_shift_30_mask_0.2_2023-03-25_90.txt` or can be found at this link https://zenodo.org/record/8170005 

  

The download link should take to a page that should look like this 

  

  

![Screenshot from 2023-07-20 18-14-19](https://github.com/gravelCompBio/Phos-ST-temp/assets/75225868/109c898e-49cc-4849-abb6-1dcb1f3aa5c1) 

  

Click the download box highlighted in picture above 

  

</br> 

  

### After picking one of the options above to download the training weights see below 

  

  

Once downloaded, **unizip** the folder and place in the `Phosformer-ST` along with all the other files in this github repository 

  

The final `Phosformer-ST` directory orinization should have the following files/folder  

  

- file 1 `phos-ST_Example_Code.ipynb` 

  

- file 2 `modeling_esm.py` 

   

- file 3 `configuration_esm.py` 

  

- file 4 `tokenization_esm.py` 

  

- file 5 `multitask_MHA_esm2_t30_150M_UR50D_neg_ratio_8+8_shift_30_mask_0.2_2023-03-25_90.txt` 

  

- file 6 `phosST.yml` 

   

- file 7 `Readme.md`



- file 8 `LICENSE`

  

- folder 1 `multitask_MHA_esm2_t30_150M_UR50D_neg_ratio_8+8_shift_30_mask_0.2_2023-03-25_90` (make sure it is unzipped) 

  

:tada: Once you have a folder with the files/folder above you have all the required files to run the model :tada: 

  

  

</br> 

  

</br> 

  

   

  

## ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) Installing dependencies with conda  

  

### PICK ONE of the options below  

### Main Option) Utilizing the PhosformerST.yml file 

here is a step-by-step guide to set up the environment with the yml file  

  

Just type these lines of code into the terminal after you download this repository (this assumes you have anaconda already installed) 

  

```   
conda env create -f phosST.yml -n PhosST  
```   

```   
conda deactivate 
```   

```   
conda activate phosST  
```   

  

### Alternative option) Creating this environment without yml file 

(This is if torch is not working with your version of cuda or any other problem) 

Just type these lines of code into the terminal after you download this repository (this assumes you have anaconda already installed) 

```   
conda create -n phosST python=3.9  
``` 

```   
conda deactivate 
``` 

```   
conda activate phosST  
``` 

```   
conda install -c conda-forge jupyterlab 
``` 

```   
pip3 install numpy==1.24.3 
``` 

```   
pip3 install pandas==2.0.2 
``` 

```   
pip3 install matplotlib==3.7.1 
``` 

```   
pip3 install scikit-learn==1.2.2 
``` 

```   
pip3 install tqdm==4.65.0 
``` 

```   
pip3 install fair-esm==2.0.0 
``` 

```   
pip3 install transformers==4.31.0 
``` 

### **For torch you will have to download to the torch's specification if you want gpu acceleration from this website** https://pytorch.org/get-started/locally/ 

  

```   
pip3 install torch torchvision torchaudio 
``` 

  

### the terminal line above might look different for you  

  

We provided code to test Phosformer-ST (see section below) 

  

  

</br> 

  

</br> 

  

  

  

## Utilizing the Model with our example code 

All the following code examples is done inside of the `phos-ST_Example_Code.ipynb` file using jupyter lab 

  

Once you have your environment resolved just use jupyter lab to access the example code by typing the command below in your terminal (when you're in the `Phosformer-ST` folder)  

```   

jupyter lab 

``` 

Once you open the notebook on your browser, run each cell in the notebook  

  

</br> 

  

### Testing Phosformer-ST with the example code 

There should be a positive control and a negative control example code at the bottom of the `phos-ST_Example_Code.ipynb` file which can be used to test the model. 
  

**Positive Example** 

```Python 

# P17612 KAPCA_HUMAN 

kinDomain="FERIKTLGTGSFGRVMLVKHKETGNHYAMKILDKQKVVKLKQIEHTLNEKRILQAVNFPFLVKLEFSFKDNSNLYMVMEYVPGGEMFSHLRRIGRFSEPHARFYAAQIVLTFEYLHSLDLIYRDLKPENLLIDQQGYIQVTDFGFAKRVKGRTWTLCGTPEYLAPEIILSKGYNKAVDWWALGVLIYEMAAGYPPFFADQPIQIYEKIVSGKVRFPSHFSSDLKDLLRNLLQVDLTKRFGNLKNGVNDIKNHKWF" 

# P53602_S96_LARKRRNSRDGDPLP 

substrate="LARKRRNSRDGDPLP" 

  

phosST(kinDomain,substrate).to_csv('PostiveExample.csv') 

``` 

  

  

**Negative Example** 

```Python 

# P17612 KAPCA_HUMAN 

kinDomain="FERIKTLGTGSFGRVMLVKHKETGNHYAMKILDKQKVVKLKQIEHTLNEKRILQAVNFPFLVKLEFSFKDNSNLYMVMEYVPGGEMFSHLRRIGRFSEPHARFYAAQIVLTFEYLHSLDLIYRDLKPENLLIDQQGYIQVTDFGFAKRVKGRTWTLCGTPEYLAPEIILSKGYNKAVDWWALGVLIYEMAAGYPPFFADQPIQIYEKIVSGKVRFPSHFSSDLKDLLRNLLQVDLTKRFGNLKNGVNDIKNHKWF" 

# Q01831_T169_PVEIEIETPEQAKTR 

substrate="PVEIEIETPEQAKTR" 

  

phosST(kinDomain,substrate).to_csv('NegitiveExample.csv') 

``` 

Both scores should show up in a csv file in the current directory

  

</br> 

  

### Inputting your own data for novel predictions 

One can simply take the code from above and modify the string variables `kinDomain` and `substrate` to make predictions on any given kinase substrate pairs 

  

**Formatting of the `kinDomain` and `substrate` for input for Phosformer-ST are as follows:** 

  

  - `kinDomain` should be a human Serine/Threonine kinase domain (not the full sequence).
     

  - `substrate` should be a 15mer with the center residue/char being the target Serine or Threonine being phosphorylated 

  

Not following these rules may result in dubious predictions  

  

  

</br> 

  

### How to interpret Phosformer-ST's output 

This model outputs a prediction score between 1 and 0.


We trained the model to uses a cutoff of 0.5 to distinguish positive and negative predictions 


A score of 0.5 or above indicates a positive prediction for peptide substrate phosphorylation by the given kinase

  
  

</br> 

  
  

## Troubleshooting 

  

If torch is not installing correctly or you do not have a GPU to run Phosformer-ST on, the CPU version of torch is perfectly fine to use 

  

Using the CPU version of torch might increase your run time so for large prediction datasets GPU acceleration is suggested 

  

If you just are here to test if it Phosformer-ST works, the example code should not take too much time to run on the CPU version of torch   

  

Also depending on your GPU the `batch_size` argument might need to be adjusted 


#### 2024-05-17
- if you get an 'EsmTokenizer' object has no attribute 'all_tokens' error when loading the tokenizer
- - Make sure you have version of  transformers==4.31.0 installed



### The model has been tested on the following computers with the following specifications for trouble shooting proposes 

  

</br> 

  

**Computer 1** 



NVIDIA Quadro RTX 5000 (16 GB vRAM)(CUDA Version: 12.1)  

  

Ubuntu 22.04.2 LTS 

  

Intel(R) Xeon(R) Bronze 3204 CPU @ 1.90GHz  (6 cores) x (1 thread per core) 

  

64 GB ram 



  

</br> 

  

**Computer 2** 



NVIDIA RTX A4000 (16 GB vRAM)(CUDA Version: 12.2)  

  

Ubuntu 20.04.6 LTS 

  

Intel(R) Xeon(R) Bronze 3204 CPU @ 1.90GHz  (6 cores) x (1 thread per core) 

  

64 GB ram 

  





</br> 


## Other accessory tools and resources
A webtool for Phosformer-ST can be accessed from: https://phosformer.netlify.app/. A huggingface repository can be downloaded from: https://huggingface.co/gravelcompbio/Phosformer-ST_with_trainging_weights. A huggingface spaces app is available at: https://huggingface.co/spaces/gravelcompbio/Phosformer-ST


  

 

 
