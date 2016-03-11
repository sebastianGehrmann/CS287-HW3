-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require("xlua")
require 'hdf5'

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
cmd:option('-K', 32, 'Size of samples for NCE')

-- ...



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
    model = trainNN(mlp, criterion, X, y, vX, vy, vs, tX, ts, wordEmbeddings)

end



function trainNN(model, criterion, X, y, vX, vy, vs, tX, ts, lt)    

    print(X:size(1), "size of the test set")
    --SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
    for i=1, opt.epochs do
		--shuffle data
		shuffle = torch.randperm(X:size(1))
		losstotal = 0
		--mini batches, yay
		for t=1, X:size(1), opt.batchsize do
			 --xlua.progress(t, X:size(1))

			local inputs = torch.Tensor(opt.batchsize, X:size(2)):cuda()
			local targets = torch.Tensor(opt.batchsize):cuda()
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
		--renorm
		wordEmbeddings.weight:renorm(2,2,1)
		print("\nepoch " .. i .. ", loss: " .. losstotal*opt.batchsize/X:size(1))

		yhat = model:forward(vX)
		loss, examples = criterion:forward(yhat,vy)
		perplexity = torch.exp(loss)

		print(perplexity, "Perplexity on validation set")
		print(examples, "Number examples")

		--compute perplexity on subset
		predictions = torch.DoubleTensor(vy:size(1), vs:size(2)):cuda():fill(0)
		for row=1,  vX:size(1) do
			for p=1, vs:size(2) do
				predictions[row][p] = yhat[row][vs[row][p]]
			end
		end	
		predictions = nn.SoftMax():cuda():forward(predictions)
		loss, examples = criterion:forward(predictions,vy)
		perplexity = torch.exp(loss)

		print(perplexity, "Perplexity on validation set")
		print(examples, "Number examples")


		if opt.savePreds == 'true' then
			preds = model:forward(tX)
			subpreds = torch.Tensor(preds:size(1), ts:size(2)):fill(0):cuda()
			for row=1, preds:size(1) do
				for class=1, ts:size(2) do
					cpred = preds[row][ts[row][class]]
					subpreds[row][class] = cpred
				end
			end
	    	renormalized = nn.SoftMax():cuda():forward(subpreds)
	    	val = string.format("%.2f", perplexity)
	        l = string.format("%.4f", loss)
	    	-- filename = opt.savefolder .. i .. "-" .. tostring(tX:size(2)+1) .. "-" .. val .. "-" .. l .. ".txt"
	     	-- torch.save(filename, renormalized, 'ascii')
	    	filename = opt.savefolder .. i .. "-" .. tostring(tX:size(2)+1) .. "-" .. val .. "-" .. l .. ".h5"
			local myFile = hdf5.open(filename, 'w')
			myFile:write('preds', renormalized:float())
			myFile:close()
    	end
	end

   return model
end

function nce(X, y, vX, vy, vs, dwin, nclasses, tX, ts)
	

	--transform input so that lookuptable is position sensitive
	if dwin==2 then
		X:narrow(2,2,1):add(nclasses)
		vX:narrow(2,2,1):add(nclasses)
	end
	

	--Normal NNLM!
	local mlp = nn.Sequential()
	--embeddings 
	local wordEmbeddings = nn.LookupTable(dwin*nclasses, opt.demb)
    local reshapeEmbeddings = nn.Reshape(dwin*opt.demb)
    mlp:add(wordEmbeddings):add(reshapeEmbeddings)

    --non-linearity
    local lin1Layer = nn.Linear(dwin*opt.demb, opt.dhid)
    local tanhLayer = nn.Tanh()
	mlp:add(lin1Layer):add(tanhLayer)

	--scoring and output embeddings / HERE IS THE SHARED STUFF WITH NCE
    local linear = nn.Linear(opt.dhid, nclasses)
    mlp:add(linear)
    --force distribution
    mlp:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()

    --NCE!!
    local nce = nn.Sequential()
    nce:add(wordEmbeddings):add(reshapeEmbeddings)
    nce:add(lin1Layer):add(tanhLayer)
    --HERE IS THE DIFFERENCE
    local sublinear = nn.Linear(opt.dhid, opt.batchsize + opt.K)
    nce:add(sublinear)


    if opt.gpuid >= 0 then
      mlp:cuda()
      criterion:cuda()
      nce:cuda()
    end

    ----TRAINING--------------------------------------------------------------------------
	--for the log probs
	wordprobs = wordprob(y, true)
    print(X:size(1), "size of the test set")
    --SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
    for i=1, opt.epochs do
		--shuffle data
		shuffle = torch.randperm(X:size(1))
		losstotal = 0
		--mini batches, yay
		for t=1, X:size(1), opt.batchsize do
			 xlua.progress(t, X:size(1))

			local inputs = torch.Tensor(opt.batchsize, X:size(2)):cuda()
			local targets = torch.Tensor(opt.batchsize+opt.K):cuda()
			local k = 1
			--only compute update when there are enough samples left
			if t+opt.batchsize+opt.K-1 > X:size(1) then break end

			for i = t,t+opt.batchsize-1 do
				inputs[k] = X[shuffle[i]]
				targets[k] = y[shuffle[i]]
				--subset sublinear
				sublinear.weight[k] = linear.weight[y[i]]
				sublinear.bias[k] = linear.bias[y[i]]
				k = k+1
			end
			--subset sublinear
			k=opt.batchsize
			for i =t+opt.batchsize, t+opt.batchsize+opt.K-1 do
				targets[k] = y[shuffle[i]]
				sublinear.weight[k] = linear.weight[y[i]]
				sublinear.bias[k] = linear.bias[y[i]]
				k = k+1
			end
			
			--zero out
			mlp:zeroGradParameters()	
			nce:zeroGradParameters()	
			--predict and compute loss
			preds = nce:forward(inputs)
			--DIFFERENT HERE - TO DO
			-- """
			-- To do backprop you then need to create a vector 
			-- deriv to feedback derivative for each true word 
			-- and the samples at each input. You can do this 
			-- by making a little scalar network that takes 
			-- each of these values, subtracts the K log prob, 
			-- applies a sigmoid, and then an NLL criterion. 
			-- """
			--HERE IS THIS APPROACH
			predloss = torch.Tensor(opt.batchsize, 1+opt.K):cuda():fill(0)
			for cpred=1, opt.batchsize do
				--takes each of these values
				--subtracts the k log prob
				predloss[cpred][1] = preds[cpred][cpred] - opt.K * wordprobs[targets[cpred]]
				
				for csample=1, opt.K do
					predloss[cpred][1+csample] = preds[cpred][opt.batchsize+csample] - opt.K * wordprobs[targets[opt.batchsize+csample]]			
				end
			end
			--applies a sigmoid
			predloss = nn.Sigmoid():cuda():forward(predloss)
			--add in all the zeros again and construct targets
			newTargets = torch.Tensor(opt.batchsize):cuda():fill(0)
			predArray = torch.Tensor(opt.batchsize, opt.batchsize+opt.K):cuda()
			for cpred=1, opt.batchsize do
				for csample=1, opt.batchsize+opt.K do
					if cpred == csample or csample > opt.batchsize then
						if cpred == csample then
							predArray[cpred][csample]=predloss[cpred][1]
						else
							predArray[cpred][csample]=predloss[cpred][1+csample-opt.batchsize]
						end
					else
						predArray[cpred][csample]=0
					end
				end
				newTargets[cpred] = cpred
			end

			loss = criterion:forward(predArray, newTargets) 
			dLdpreds = criterion:backward(predArray, newTargets)


			losstotal = losstotal + loss     
			nce:backward(inputs, dLdpreds)
			nce:updateParameters(opt.eta)

			--COPY BACK THE WEIGHTS
			k=1
			for i = t, t + opt.batchsize + opt.K -1 do
				linear.weight[y[i]] = sublinear.weight[k]
        		linear.bias[y[i]] = sublinear.bias[k]
        		k = k+1
			end

		end
		--renorm
		wordEmbeddings.weight:renorm(2,2,1)
		print("\nepoch " .. i .. ", loss: " .. losstotal*opt.batchsize/X:size(1))

		yhat = mlp:forward(vX)
		loss, examples = criterion:forward(yhat,vy)
		perplexity = torch.exp(loss)

		print(perplexity, "Perplexity on validation set")
		print(examples, "Number examples")

		--compute perplexity on subset
		predictions = torch.DoubleTensor(vy:size(1), vs:size(2)):cuda():fill(0)
		for row=1,  vX:size(1) do
			for p=1, vs:size(2) do
				predictions[row][p] = yhat[row][vs[row][p]]
			end
		end	
		predictions = nn.SoftMax():forward(predictions)
		loss, examples = criterion:forward(predictions,vy)
		perplexity = torch.exp(loss)

		print(perplexity, "Perplexity on validation set")
		print(examples, "Number examples")
	end

end

function wordprob(y, logs)
	probs = {}
	setDefault(probs, 0)
	for row=1, y:size(1) do
		probs[y[row]] = probs[y[row]] + 1
	end
	for row=1, y:size(1) do
		probs[y[row]] = probs[y[row]]/y:size(1)
		if logs then
			if not probs[y[row]] == 0 then
				probs[y[row]] = torch.log(probs[y[row]])
			end
		end
	end

	return probs
end

function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	local f = hdf5.open(opt.datafile, 'r')

	if opt.gpuid >= 0 then
	  print('using CUDA on GPU ' .. opt.gpuid .. '...')
	  require 'cutorch'
	  require 'cunn'
	  cutorch.setDevice(opt.gpuid + 1)
	end


	nclasses = f:read('nclasses'):all():long()[1]
	nfeatures = f:read('nfeatures'):all():long()[1]
	dwin = f:read('dwin'):all():long()[1]

	tin = f:read('train_input'):all():cuda()
	tout = f:read('train_output'):all():squeeze():cuda()

	vin = f:read('valid_blanks_input'):all():cuda()
	vset = f:read('valid_blanks_set'):all():cuda()
	vset = vset:narrow(2, dwin+1, vset:size(2)-dwin-1):cuda()
	vout = f:read('valid_output'):all():squeeze():cuda()

	testin = f:read('test_input'):all():cuda()
	testout = f:read('test_set'):all():cuda()
	testout = testout:narrow(2, dwin+1, testout:size(2)-dwin-1):cuda()

	if opt.dev == 'true' then
      print('Development mode')
      print('Narrowing the Training Data to 100 Samples')
      tin = tin:narrow(1, 1, 1000):clone()
      tout = tout:narrow(1, 1, 1000):clone()
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
	elseif opt.lm == "nce" then
		nce(tin, tout, vin, vout, vset, dwin, nclasses, testin, testout)
	end
	-- Test.
	-- for test we have to renormalize the scores of the given words
end

main()
