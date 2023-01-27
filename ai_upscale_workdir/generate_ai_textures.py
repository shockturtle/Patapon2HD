import os
import shutil
from pathlib import Path

import upscale

processAlpha = None

inputVal=input("Need process alpha channel ? (Y,n)")
if inputVal.lower() == "n":
    processAlpha = False
elif inputVal.lower() == "y" or not inputVal:
    processAlpha = True
else:
    print("Error input")
    exit(0)

print("Select upscale engine, or if you want use multiple, write multiple numbers like 123")

models = os.listdir("models")

if len(models) == 0:
    print("Download models and place in it models before use")
    exit(0)

for i in range(len(models)):
    print(str(i + 1) + ". " + models[i])

select = input("Select: ")
if not select.isdigit():
    print("Write digit")
    exit(0)

selectedModels = []

for i in select:
    num = int(i) - 1
    if num < 0 or num >= len(models):
        print("Number bigger of models size or lower 0")
        exit(0)
    else:
        selectedModels.append(models[num])

print("Start: " + "    ->     ".join(selectedModels))


def deleteFolder(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


tempFolder = "TEMP"
finalOutputFolder = "output"

deleteFolder(tempFolder)

inputFolder = "input"

for i in range(len(selectedModels)):
    model = selectedModels[i]
    modelResultFolder = "TEMP" + os.sep + str(i + 1) + "(" + model[0:20] + ")"

    upscale.Upscale(
        model=model,
        input=Path(inputFolder),
        output=Path(modelResultFolder),
        alpha_mode=upscale.AlphaOptions.alpha_separately if processAlpha else upscale.AlphaOptions.no_alpha
    ).run()

    inputFolder = modelResultFolder

deleteFolder(finalOutputFolder)
shutil.copytree(inputFolder, finalOutputFolder)
