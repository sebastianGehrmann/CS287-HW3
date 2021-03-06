import numpy as np
import pandas as pd
import os
import h5py
for i, f in enumerate(os.listdir("predictions")):	
	if ".h5" in f:
		myFile = h5py.File("predictions/"+f, 'r')
		data = myFile['preds'][:]
		df = pd.DataFrame(data)
		print(f)
		#print(df.shape)
		df.columns = ["Class"+str(i+1) for i in range(50)]
		ids = [i+1 for i in range(len(data))]
		df.insert(0, 'ID',ids)
		df.to_csv('kaggle/'+f[:-3]+'.csv', index=False)