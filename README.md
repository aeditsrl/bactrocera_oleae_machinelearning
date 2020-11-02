# **Machine learning to predict _Bactrocera oleae_ infestation**
This repository contains a script and an example dataset for the analysis carried out to write the paper “Managing complex datasets to predict *Bactrocera oleae* infestation at the regional scale” authored by Iride Volpi, Diego Guidotti, Michele Mammini, Ruggero Petacchi and Susanna Marchi.
The paper was submitted to the journal Computers and Electronics in Agriculture on 8 June 2020.

## Description of the files
# R script
The R script contains all the steps of the methodology adopted in the paper to predict the presence of infestation of *B. oleae* in Tuscany Region (Italy) with machine learning using a set of variables associated with the infestation (i.e. bioclimatic indices, geographical indices and the infestation of the previous year):
* Dataset partition
* Selection of variables
* Training and selection of the classifiers
* Predictions on the test sets
* Variable importance and partial dependence plots

# Example dataset
A subset of the entire dataset used in the paper was provided. The dataset contains observation of the infestations for a set of olive groves (farm_ID) and the indices associated with the infestation.
The indices were calculated according to what reported in the section 2.2 of the paper.
