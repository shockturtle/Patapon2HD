import os

files = os.listdir("./textures/ai")

for file in files:
    print(file + " = textures/ai/" + file)