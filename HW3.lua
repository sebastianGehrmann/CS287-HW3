-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require("xlua")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-lm', 'mle', 'classifier to use')
cmd:option('-dev', 'false', 'narrow training data for development')
cmd:option('-savePreds', 'false', 'Save the predictions of testset')
cmd:option('-gpuid', -1, 'set to >=0 for cuda')
cmd:option('-savefolder', 'predictions/','filename to autosave the checkpont to')

-- Hyperparameters
cmd:option('-alpha', 0, 'laplace smoothing value')
cmd:option('-dhid', 100, 'Number of hidden units')
cmd:option('-demb', 50, 'Size of the embedding for Words')
cmd:option('-eta', .02, 'Learning Rate')
cmd:option('-epochs', 20, 'Number of Epochs')
cmd:option('-batchsize', 32, 'Size of the Minibatch for SGD')

-- ...

-- helper functions
function setDefault (t, d)
	--sets default value for table
 	local mt = {__index = function () return d end}
    setmetatable(t, mt)
end

function perplexity(yhat, y)
	--too low right now?!
	criterion = nn.ClassNLLCriterion()
	err = criterion:forward(yhat, y)
	err = torch.exp(err)
	--print("\n")
	print("\nPerplexity: ", err)
end


function maximumLikelihoodTable(X, y, alpha, dwin, nclasses)
	local Fc = {}
	setDefault(Fc, 0) --laplace
	local Fcw = {}
	setDefault(Fcw, 0) --laplace
	print("\nCompute MLE Table")
	for row=1, X:size(1) do
		if row%1000 == 0 then
			xlua.progress(row, X:size(1))
		end
		if dwin==1 then
			setDefault(Fcw, alpha*nclasses)
			if Fcw[X[row][1]] == 0 then
				c = {}
				setDefault(c, alpha) --laplace
				Fcw[X[row][1]] = c
			end
			Fc[X[row][1]] = Fc[X[row][1]] + 1
			Fcw[X[row][1]][y[row][1]] = Fcw[X[row][1]][y[row][1]] + 1
		elseif dwin==2 then
			--deal with empty dicts
			if Fc[X[row][1]] == 0 then
				c = {}
				setDefault(c, alpha*nclasses) --laplace
				Fc[X[row][1]] = c
			end
			if Fcw[X[row][1]] == 0 then
				c = {}
				setDefault(c, 0) --laplace
				Fcw[X[row][1]] = c
			end
			if Fcw[X[row][1]][X[row][2]] == 0 then
				c = {}
				setDefault(c, alpha) --laplace
				Fcw[X[row][1]][X[row][2]] = c
			end
			-- count the context
			Fc[X[row][1]][X[row][2]] = Fc[X[row][1]][X[row][2]] + 1
			Fcw[X[row][1]][X[row][2]][y[row][1]] = Fcw[X[row][1]][X[row][2]][y[row][1]] + 1

		else 
			print("longer n-grams not implemented")
		end
	end
	return Fcw, Fc
end

function MLE(X, y, vX, vy, vs, alpha, dwin, nclasses)
	--p(w|c) = F(w,c)/F(c)
	Fcw, Fc = maximumLikelihoodTable(X, y, alpha, dwin, nclasses)
	print("\nLikelihood table constructed")
	--for every example take max likelihood / compute perplexity
	--only test on blanks? Store table in between?
	predictions = torch.DoubleTensor(vy:size(1), vs:size(2)):fill(0)
	
	print("Prediction Process:")
	for row=1, vy:size(1) do
		xlua.progress(row, vy:size(1))
		if dwin==1 then
			for p=1,vs:size(2) do
				if Fc[vX[row][1]] == 0 then
					predictions[row][p] = 0
				else	
					norm_factor = nclasses/vs:size(2)
					predictions[row][p] = norm_factor * Fcw[vX[row][1]][vs[row][p]]/Fc[vX[row][1]]
				end
			end
		elseif dwin==2 then
			--for all words to predict
			for p=1,vs:size(2) do
				if Fc[vX[row][1]] == 0 then
					predictions[row][p] = 0
				elseif Fc[vX[row][1]][vX[row][2]] == alpha*nclasses then
					predictions[row][p] = 0
				else	
					norm_factor = nclasses/vs:size(2)
					predictions[row][p] = norm_factor * Fcw[vX[row][1]][vX[row][2]][vs[row][p]]/Fc[vX[row][1]][vX[row][2]]
				end
			end
		else 
			print("longer n-grams not implemented")
		end
	end
	normalization = nn.LogSoftMax()
	preds = normalization:forward(predictions)
	perplexity(preds, vy)
end

function wordprob(y)
	probs = {}
	setDefault(probs, 0)
	for row=1, y:size(1) do
		probs[y[row][1]] = probs[y[row][1]] + 1
	end
	return probs
end

function wittenBell(X, y, vX, vy, vs, dwin, nclasses)
	--To Do: Adjust this for trigrams
	local Fcw, Fc = maximumLikelihoodTable(X, y, 0, dwin, nclasses)
	local Fccw = {}
	local Fcc = {}
	if dwin==2 then
		Fccw, Fcc = maximumLikelihoodTable(X:narrow(2, 2, X:size(2)-1), y, 0, dwin-1, nclasses)
	end
	Fw  = wordprob(y)
	print("Likelihood table constructed")
	predictions = torch.DoubleTensor(vy:size(1), vs:size(2)):fill(0)
	print("Prediction Process")
	for row=1, 100 do--vy:size(1) do
		xlua.progress(row, vy:size(1))
		if dwin==1 then
			for p=1,vs:size(2) do
				if Fc[vX[row][1]] == 0 then
					predictions[row][p] = 0
				else	
					fc = Fc[vX[row][1]]
					nc = #Fcw[vX[row][1]]
					fcw = Fcw[vX[row][1]][vs[row][p]]
					fw = Fw[vs[row][p]]/vy:size(1)
					norm_factor = nclasses/vs:size(2)
					predictions[row][p] = norm_factor * (fcw+nc*fw)/(fc+nc) 
				end
			end
		elseif dwin==2 then
			-- 2) implement formula 
			for p=1,vs:size(2) do
				if Fc[vX[row][1]] == 0 then
					predictions[row][p] = 0
				elseif Fc[vX[row][1]][vX[row][2]] == 0 then
					predictions[row][p] = 0
				else	
					--word prob
					fw = Fw[vs[row][p]]/vy:size(1)
					--bigram prob
					fcw = Fccw[vX[row][2]][vs[row][p]]
					nc = #Fccw[vX[row][2]]
					fc = Fcc[vX[row][2]]
					pwb = (fcw+nc*fw)/(fc+nc) 
					--trigram prob
					fcw = Fcw[vX[row][1]][vX[row][2]][vs[row][p]]
					nc = #Fcw[vX[row][1]][vX[row][2]]
					fc = Fc[vX[row][1]][vX[row][2]]
					pwb = (fcw+nc*pwb)/(fc+nc) 
					--put it all together
					norm_factor = nclasses/vs:size(2)
					predictions[row][p] = norm_factor * pwb

				end
			end
		else 
			print("longer n-grams not implemented")		
		end
	end
	normalization = nn.LogSoftMax()
	preds = normalization:forward(predictions)
	perplexity(preds, vy)

end

function nnlm(X, y, vX, vy, vs, dwin, nclasses, tX, ts)
	--transform input so that lookuptable is position sensitive
	if dwin==2 then
		X:narrow(2,2,1):add(nclasses)
		vX:narrow(2,2,1):add(nclasses)
	end
	
	mlp = nn.Sequential()
	--embeddings 
	wordEmbeddings = nn.LookupTable(dwin*nclasses, opt.demb)
    reshapeEmbeddings = nn.Reshape(dwin*opt.demb)
    mlp:add(wordEmbeddings):add(reshapeEmbeddings)

    --non-linearity
    lin1Layer = nn.Linear(dwin*opt.demb, opt.dhid)
    tanhLayer = nn.Tanh()
	mlp:add(lin1Layer):add(tanhLayer)

	--scoring and output embeddings
    lin2Layer = nn.Linear(opt.dhid, nclasses)
    mlp:add(lin2Layer)
    --force distribution
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    -- loss, count = criterion:forward(mlp:forward(X), y)
    -- print(torch.exp(loss))
    --print(mlp:forward(X))
    if opt.gpuid >= 0 then
      mlp:cuda()
      criterion:cuda()
    end
    model = trainNN(mlp, criterion, X, y, vX, vy, tX, ts)

end



function trainNN(model, criterion, X, y, vX, vy, tX, ts)    

   print(X:size(1), "size of the test set")
   --SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
   for i=1, opt.epochs do
      --shuffle data
      shuffle = torch.randperm(X:size(1))
      losstotal = 0
      --mini batches, yay
      for t=1, X:size(1), opt.batchsize do
         xlua.progress(t, X:size(1))

         local inputs = torch.Tensor(opt.batchsize, X:size(2))
         local targets = torch.Tensor(opt.batchsize)
         local k = 1
         for i = t,math.min(t+opt.batchsize-1,X:size(1)) do
            -- load new sample
            inputs[k] = X[shuffle[i]]
            targets[k] = y[shuffle[i]]
            k = k+1
         end
         k=k-1
         --in case the last batch is < batchsize
         if k < opt.batchsize then
           inputs = inputs:narrow(1, 1, k):clone()
           targets = targets:narrow(1, 1, k):clone()
         end
         --zero out
         model:zeroGradParameters()
         --predict and compute loss
         preds = model:forward(inputs)
         loss = criterion:forward(preds, targets) 
         losstotal = losstotal + loss     
         dLdpreds = criterion:backward(preds, targets)
         model:backward(inputs, dLdpreds)
         model:updateParameters(opt.eta)
      end
      print("\nepoch " .. i .. ", loss: " .. losstotal*opt.batchsize/X:size(1))

      yhat = model:forward(vX)
      loss, examples = criterion:forward(yhat,vy)
      perplexity = torch.exp(loss)

      print(perplexity, "Perplexity on validation set")

      if opt.savePreds == 'true' then
    	preds = model:forward(tX)
    	subpreds = torch.Tensor(preds:size(1), ts:size(2)):fill(0)
    	for row=1, preds:size(1) do
    		for class=1, ts:size(2) do
    			cpred = preds[row][ts[row][class]]
    			subpreds[row][class] = cpred
    		end
    	end
    	renormalized = nn.SoftMax():forward(subpreds)
    	val = string.format("%.2f", perplexity)
        l = string.format("%.4f", loss)
    	filename = opt.savefolder .. i .. "-" .. tostring(tX:size(2)+1) .. "-" .. val .. "-" .. l .. ".txt"
        torch.save(filename, renormalized, 'ascii')
    end
   end

   return model
end

function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	local f = hdf5.open(opt.datafile, 'r')
	nclasses = f:read('nclasses'):all():long()[1]
	nfeatures = f:read('nfeatures'):all():long()[1]
	dwin = f:read('dwin'):all():long()[1]

	tin = f:read('train_input'):all()
	tout = f:read('train_output'):all():squeeze()

	vin = f:read('valid_blanks_input'):all()
	vset = f:read('valid_blanks_set'):all()
	vset = vset:narrow(2, 2, vset:size(2)-2)
	vout = f:read('valid_output'):all():squeeze()

	testin = f:read('test_input'):all()
	testout = f:read('test_set'):all()

	if opt.dev == 'true' then
      print('Development mode')
      print('Narrowing the Training Data to 100 Samples')
      tin = tin:narrow(1, 1, 1000):clone()
      tout = tout:narrow(1, 1, 1000):clone()
   end

   if opt.gpuid >= 0 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpuid + 1)
   end

	-- Train.
	if opt.lm == "mle" then
		MLE(tin, tout, vin, vout, vset, 0, dwin, nclasses)
	elseif opt.lm == "laplace" then
		MLE(tin, tout, vin, vout, vset, opt.alpha, dwin, nclasses)		
	elseif opt.lm == "wb" then
		wittenBell(tin, tout, vin, vout, vset, dwin, nclasses)
	elseif opt.lm == "nn" then
		nnlm(tin, tout, vin, vout, vset, dwin, nclasses, testin, testout)
	end
	-- Test.
	-- for test we have to renormalize the scores of the given words
end

main()
