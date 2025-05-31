@echo off
echo Downloading T2M evaluators
gdown --fuzzy https://drive.google.com/file/d/1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb/view
gdown --fuzzy https://drive.google.com/file/d/12liZW5iyvoybXD8eOw4VanTgsMtynCuU/view
if exist t2m (
    rmdir /s /q t2m
)
if exist kit (
    rmdir /s /q kit
)

echo Cleaning
del t2m.zip
del kit.zip

echo Downloading done!
