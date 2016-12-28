local THNN = require 'nn.THNN'
npy4th = require 'npy4th'
local LinearSparse, parent = torch.class('nn.LinearSparse', 'nn.Module')

function CreateRandomSparseWeightMatrix(density, inputSize, outputSize)
  randElements = torch.randperm(inputSize*outputSize)

  nElements = math.floor(density * inputSize* outputSize)

  weights = torch.Tensor(nElements)
  rows = torch.Tensor(nElements):int()
  cols = torch.Tensor(nElements):int()
  iRowStart = torch.Tensor(outputSize+1):int()

  t = 1
  for i = 0, outputSize - 1 do
    for j = 0, inputSize - 1 do
       rows[t] = i
       cols[t] = j
       t = t+1
     end
   end

 --  for i = 1,nElements do
 --   rows[i] = math.floor((randElements[i] - 1)/inputSize)
 --   cols[i] = (randElements[i] - 1) % inputSize
 -- end

 rows, i = rows:sort()
 cols = cols:index(1, i)

iRowStart[rows[1]+1] = 0
actRow = rows[1]
istart = 1

  for i=2,nElements do
    if actRow ~= rows[i] then
      subVector = cols[{{istart,i-1}}]
      cols[{{istart,i-1}}] = subVector:sort()
      iRowStart[rows[i]+1] = i-1
      actRow = rows[i]
      istart = i
    end
  end

  subVector = cols[{{istart,nElements}}]
  cols[{{istart,nElements}}] = subVector:sort()

  iRowStart[outputSize+1] = nElements
  return weights, rows, cols, iRowStart
end

function CreateAndPrepareWeights(nRows, nCols, nChannels, nChannelsOut)

  local outputSize = nCols*nRows*nChannelsOut
  os.execute("python /home/ubuntu/testsnewlayer/createSPM.py " .. nCols .. " " .. nRows .. " " .. nChannels .. " " .. nChannelsOut)
  local rows = npy4th.loadnpy("/home/ubuntu/rows.npy")
  local cols = npy4th.loadnpy("/home/ubuntu/cols.npy")

  nElements = (#rows)[1]

  local weights = torch.zeros(nElements)
  local iRowStart = torch.Tensor(outputSize+1):int()

  rows, i = rows:sort()
  cols = cols:index(1, i)

  iRowStart[rows[1]+1] = 0
  actRow = rows[1]
  istart = 1

  for i=2,nElements do
    if actRow ~= rows[i] then
      subVector = cols[{{istart,i-1}}]
      cols[{{istart,i-1}}] = subVector:sort()
      iRowStart[rows[i]+1] = i-1
      actRow = rows[i]
      istart = i
    end
  end

  subVector = cols[{{istart,nElements}}]
  cols[{{istart,nElements}}] = subVector:sort()

  iRowStart[outputSize+1] = nElements
  return weights, rows, cols, iRowStart
end

function LinearSparse:__init(imgWidth, imgHeight, nChannels, nChannelsOut, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias

   local outputSize = imgWidth*imgHeight*nChannelsOut
   self.weight, self.rows, self.cols, self.iRowStart = CreateAndPrepareWeights(imgHeight, imgWidth, nChannels, nChannelsOut)
   --self.weight, self.cols, self.iRowStart = CreateAndPrepareWeights(imgHeight, imgWidth, nChannels)
   --self.rows = torch.Tensor(1):int()
   self.nnz = self.weight:size(1)
   self.gradWeight = torch.Tensor(self.nnz)
   self.nRows = imgWidth*imgHeight*nChannelsOut
   self.nCols = imgWidth*imgHeight*nChannels

   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self:reset()
end

function LinearSparse:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.nCols)
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

function LinearSparse:updateOutput(input)

   if input:dim() == 1 then
     self.output:resize(self.nRows)
     if self.bias then self.output:copy(self.bias) else self.output:zero() end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.nRows)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
   else
      error('input must be a matrix')
   end

   input.THNN.LinearSparse_updateOutput(input:cdata(), self.output:cdata(), self.weight:cdata(), self.rows:cdata(), self.cols:cdata(), self.iRowStart:cdata() ,self.nRows, self.nCols)

   if self.bias and input:dim() == 2 then
      self.output:addr(1, self.addBuffer, self.bias)
   end

   return self.output
end

function LinearSparse:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end

      input.THNN.LinearSparse_updateGradInput(input:cdata(), gradOutput:cdata(), self.gradInput:cdata(), self.weight:cdata(), self.rows:cdata(), self.cols:cdata(), self.nRows, self.nCols, self.iRowStart:cdata())


      return self.gradInput
   end
end

function LinearSparse:accGradParameters(input, gradOutput, scale)

   scale = scale or 1
   --input.THNN.LinearSparse_accGradParameters(input:cdata(),gradOutput:cdata(), self.gradWeight:cdata(), self.rows:cdata(), self.cols:cdata(), self.nnz, scale, self.iRowStart:cdata())
   input.THNN.LinearSparse_accGradParameters(input:cdata(),gradOutput:cdata(), self.gradWeight:cdata(), self.rows:cdata(), self.cols:cdata(), self.nnz, scale)

   if input:dim() == 1 then
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      if self.bias then
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   else
     error('input must be a matrix')
   end
end

-- we do not need to accumulate parameters when sharing
LinearSparse.sharedAccUpdateGradParameters = LinearSparse.accUpdateGradParameters


function LinearSparse:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.nCols, self.nRows) ..
      (self.bias == nil and ' without bias' or '')
end
