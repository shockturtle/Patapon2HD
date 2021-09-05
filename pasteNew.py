import os
import glob
import shutil

files = glob.glob('new/*.png')

toAddLines = [
    file.removesuffix(".png") + " = textures/ai/" + file for file in
    [os.path.basename(x) for x in files]
]

if len(toAddLines) == 0:
    print("No new files")
    exit(0)

for file in files:
    shutil.move(file, "textures/ai")


TEX_INI = "textures.ini"

fileLines = set()

with open(TEX_INI) as texIniFile:
    for line in texIniFile:
        fileLines.add(line.removesuffix("\n"))


finalLines = set(toAddLines) - fileLines
if len(finalLines) == 0:
    print("No new lines")
    exit(0)

with open(TEX_INI, "a") as texIniFile:
    texIniFile.write("\n" + "\n".join(finalLines))

print("Added "+len(finalLines)+" lines")
