import numpy as np
import os
import h5py
for i, f in enumerate(os.listdir("predictions")):	
	if ".txt" in f:
		with open("predictions/"+f) as k:
			preds = k.readline().split()
			print(len(preds))
	if ".h5" in f:
		myFile = h5py.File("predictions/"+f, 'r')
		data = myFile['preds'][:]
		print(len(data[1]))
	if i>0:
		break