### Source code for submission paper "It's a Kind of Magic: A Novel Conditional GAN Framework for Efficient Profiling Side-channel Analysis"

## Run CGAN-SCA with a (no-code) webapp:
To run the locaweb app to run our CGAN-SCA solution:

```
pip install -r requirements.txt
streamlit run .\app_main.py
```

The Streamlit app already downloads all datasets and has a no-code user interface.

## Run CGAN-SCA from code:
To run a CGAN-SCA training with a MLP-based profiling attack, run the following command:

```
python main.py --dataset_reference ascad-variable --dataset_reference_dim 25000 --dataset_target ASCAD --dataset_target_dim 10000 --n_profiling_reference 200000 --n_profiling_target 50000 --epochs 200 --target_byte_reference 2 --target_byte_target 2 --features 100
```

Check ```main.py``` for more information about command line arguments.


To run an example that includes all methods and without calling the arguments from command line:

```
python example_ascad-variable_vs_ASCAD.py
```

This example will train a CGAN-SCA structure with ASCADr as reference and ASCADf as target dataset.

#### Datasets ####
To generate datasets (NOPOI scenario from [1]) allowing reproducibility of our results, please check the available source code from https://github.com/AISyLab/feature_selection_dlsca.

[1] Guilherme Perin, Lichao Wu, Stjepan Picek, "Exploring Feature Selection Scenarios for Deep Learning-based Side-Channel Analysis" (https://tches.iacr.org/index.php/TCHES/article/view/9842).