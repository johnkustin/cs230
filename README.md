# cs230
deepest of learns

### downloading the sample dataset
1. follow installation guide for [Kaggle API](https://github.com/Kaggle/kaggle-api)
2. go to the datasets directory `cd cs230/datasets/` and run `sh download-sample.sh`. if you cant execute it do `chmod +x download-sample.sh`
3. make a directory for the pictures in `/datasets/` using `mkdir faces-sample`
3. unzip the sample dataset `unzip deepfake-detection-faces-sample.zip -d faces-sample/` this will take a few minutes

### baseline model

Taken from the team "The Medics" who competed on the Deepfake Detection Challenge (DFDC) on Kaggle.<br/>
> Notebook name: DFDC 3D & 2D inc cutmix with 3D model fix<br/>
> Team: The Medics. (Ian Pan & James Howard) <br/>
> Private score (log loss): 0.43711<br/>
> Public score (log loss): 0.25322<br/>
> [Kaggle notebook available here](https://www.kaggle.com/vaillant/dfdc-3d-2d-inc-cutmix-with-3d-model-fix)<br/>
