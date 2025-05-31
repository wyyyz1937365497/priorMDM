@echo off
echo Downloading glove (in use by the evaluators, not by MDM itself)
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
if exist glove (
    rmdir /s /q glove
)

echo Cleaning
del glove.zip

echo Downloading done!
