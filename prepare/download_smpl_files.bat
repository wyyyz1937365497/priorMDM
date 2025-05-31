@echo off
mkdir body_models
cd body_models

echo The smpl files will be stored in the 'body_models/smpl/' folder
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
if exist smpl (
    rmdir /s /q smpl
)

echo Cleaning
del smpl.zip

echo Downloading done!
