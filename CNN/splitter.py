import os
import shutil
from pathlib import Path

def make_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)

def move_to_directory(file, start_path, end_path):
	for line in file.readlines():
		line = line.rstrip('\n')
		shutil.copyfile(start_path + line , end_path + "/" + line )
		
def move_to_train(file, start_path, end_path):
	i = 0
	for line in file.readlines():
		print(i)
		if i%2 == 0:
			i+=1
			print("skipping", line)
			continue
			
		line = line.rstrip('\n')
		shutil.move(start_path + line , end_path + "/" + line )
		i+=1


nr = 1

make_dir("./dataset")
make_dir("./dataset/train")
make_dir("./dataset/validation")
make_dir("./dataset/test")

for dir in os.listdir("./dtd-r1.0.1/dtd/images"):
	make_dir("./dataset/train/" + dir)
	make_dir("./dataset/validation/" + dir)
	make_dir("./dataset/test/" + dir)

train = open("./dtd-r1.0.1/dtd/labels/train" + str(nr) + ".txt", "r") 
validation = open("./dtd-r1.0.1/dtd/labels/val" + str(nr) + ".txt", "r") 
test = open("./dtd-r1.0.1/dtd/labels/test" + str(nr) + ".txt", "r") 

move_to_directory (train, "./dtd-r1.0.1/dtd/images/" , "./dataset/train/")
move_to_directory(validation,"./dtd-r1.0.1/dtd/images/" , "./dataset/validation/")
move_to_directory(test, "./dtd-r1.0.1/dtd/images/" , "./dataset/test/")

validation = open("./dtd-r1.0.1/dtd/labels/val" + str(nr) + ".txt", "r") 
test = open("./dtd-r1.0.1/dtd/labels/test" + str(nr) + ".txt", "r") 

move_to_train(validation, "./dataset/validation/" , "./dataset/train/")
move_to_train(test, "./dataset/test/" , "./dataset/train/")






   