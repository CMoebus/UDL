### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 64c51f97-6b98-4253-8a89-3cb22c9c80a5
begin
	#-------------------------------------------------------------------------
	# Neural Nets
	using Lux, Optimisers, Random
	using ADTypes, Zygote
	#-------------------------------------------------------------------------
	# Output
	using Plots
	# using PlutoUI
	# using Printf
	#-------------------------------------------------------------------------
	# Statistics
	using Statistics
	#-------------------------------------------------------------------------
	# using Logistics
	# using SpecialFunctions
	#-------------------------------------------------------------------------
	# LateXify Strings
	using LaTeXStrings, Latexify
	#-------------------------------------------------------------------------
end # begin

# ╔═╡ 08e3a970-3ac9-11ef-30de-ad60816a0a04
md"
=====================================================================================
#### UDL\_20240705\_5\_1\_Identification\_Of\_A\_Latent\_Generator.jl
##### file:  UDL_20240705_5_1_IdentificationOfALatentGenerator.jl
##### code: Julia 1.10.4/Pluto by *** PCM 2024/07/11 ***

=====================================================================================
"

# ╔═╡ 73d1e53f-1090-42c0-b76d-28242f907ffd
md"
##### 0. Introduction
"

# ╔═╡ be780bbe-645d-4342-8020-4a33704dd6a4
md"
---
##### 1. Packages
"

# ╔═╡ 5e4dd047-f1fb-4b3a-8d7c-e320b9821116
md"
---
##### 2. Specification of Shallow 1-3-1-Generator
###### 2.1 General Matrix Formulation
(Prince, 2024, (4.15))

$\mathbf{h_1} = \mathbf a \left(\mathbf{\beta}_0 + \mathbf{\Omega}_0 \mathbf x \right)$

$\;$

$...$

$\;$

$\mathbf{h_k} = \mathbf a \left(\mathbf{\beta}_{k-1} + \mathbf{\Omega}_{k-1} \mathbf x \right)$

$\;$

$\mathbf{y} = \mathbf a \left(\mathbf{\beta}_{k} + \mathbf{\Omega}_{k} \mathbf x \right)$

$\;$
$\;$

"

# ╔═╡ 67260b0d-7276-448b-8d5e-fbf8c8311a42
md"
---
###### 2.2 Julia-Code for Shallow 1-3-1 Model
"

# ╔═╡ deab1f59-b1a6-4d76-bbd6-97910cc04419
#-------------------------------------------------------------------------------------
# Define a shallow neural network in matrix form with 
###  - one input layer with one unit, 
###  - one hidden layer with three hidden units, 
###  - and one output layer with one unit
#-------------------------------------------------------------------------------------
function shallow_1_3_1_nn(xs::Vector{Float32}, activation::Function, 
	β0::Array{Float32}, β1::Array{Float32}, 
	Ω0::Array{Float32}, Ω1::Array{Float32})
	#---------------------------------------------------------------------------------
	# layer 1
	# preactivation 
	layer1_preact = β0 .+ (Ω0 * xs')                     # Prince, 2024, (4.15)
	# Pass these through the ReLU function 
	h1 = activation.(layer1_preact)                      # Prince, 2024, (4.15)
	#---------------------------------------------------------------------------------# weight the activations                                 # Prince, 2024, (4.15)
	ys = β1 .+ (Ω1 * h1)
	#---------------------------------------------------------------------------------
	layer1_preact, h1, ys
end # function shallow_1_3_1_nn

# ╔═╡ b7aaf6e9-3563-47f6-8983-f59479afe703
md"
---
###### 2.3 Parameters
"

# ╔═╡ d792444b-69a0-4225-95b5-c46c83ef0e89
function getParameters()
	#--------------------------------------------------------------------
	# coefficients influencing layer 1
	β0 = Array{Float32}(undef, (3, 1))
	Ω0 = Array{Float32}(undef, (3, 1))
	β0[1] =  0.3f0; β0[2] = -1.0f0; β0[3] = -0.5f0
    Ω0[1] = -1.0f0; Ω0[2] =  1.8f0; Ω0[3] =  0.65f0
	#--------------------------------------------------------------------
	# coefficients influencing output layer
	β1 = Array{Float32}(undef, (1, 1))
	Ω1 = Array{Float32}(undef, (1, 3))
	β1[1] =  0.1f0
    Ω1[1] = -2.0f0; Ω1[2] = -1.0f0; Ω1[3] = 7.0f0
	#--------------------------------------------------------------------
	β0, Ω0, β1, Ω1
end # function getParameters

# ╔═╡ c9c1d3c6-8fd7-4d7a-a173-5016b0bdb5cf
β0, Ω0, β1, Ω1 = getParameters()

# ╔═╡ 831555f2-5bdb-43cd-895a-c3a5eadb3be3
md"
---
###### 2.4. Generator Run
###### 2.4.1 Input Definition
"

# ╔═╡ 9a3bbc35-3dad-4d92-acc2-1e8dac451e81
begin
	# range of input values
	xs = convert(Vector{Float32}, [x for x in 0.0:0.01:1.0])
	const N = length(xs)
end # begin

# ╔═╡ 78258a9c-8594-437e-b0b4-cc3c29f946aa
xs

# ╔═╡ dd7122c4-2fba-46af-b560-f0649d4c95b5
size(xs), typeof(xs)

# ╔═╡ 3f78e783-d6b9-450f-9914-e21fce295a14
md"
---
###### 2.4.2 Run Generator
"

# ╔═╡ 96a59f0e-d22e-4476-b5f2-69018a9a9fb8
relu

# ╔═╡ d7643abe-a1ff-496c-86c9-e9364cb2bd92
# Run the neural network
layer1_preact, h1, ysHat = 
	shallow_1_3_1_nn(xs, relu, β0, β1, Ω0, Ω1)

# ╔═╡ 1adf5c3e-4e88-4175-9cf4-78916d3eade8
ysHat

# ╔═╡ fb2441a4-59eb-4439-9f60-3baafe33699d
let plot1  = 
		plot(xs, layer1_preact[1, :], label=L"layer1\_preact1", title=L"β_{01}+Ω_{01}")
	plot2  = 
		plot(xs, layer1_preact[2, :], label=L"layer1\_preact2", title=L"β_{02}+Ω_{02}")
	plot3  = 
		plot(xs, layer1_preact[3, :], label=L"layer1\_preact3", title=L"β_{03}+Ω_{03}")
	plot(plot1, plot2, plot3, ylimit=(-1, 1), lw=2, titlefontsize=12)
end # let

# ╔═╡ 73bb7782-8d52-4905-9c97-b7cf0fc44fe4
let plot4  = 
		plot(xs, h1[1, :], label=L"h1[1]", xlabel=L"xs", ylabel=L"h1_{1}", title=L"h_{11}=a(β_{01}+Ω_{01})")
	plot5  = 
		plot(xs, h1[2, :], label=L"h1[2]", xlabel=L"xs", ylabel=L"h1_{2}", title=L"h_{12}=a(β_{02}+Ω_{02})")
	plot6  = 
		plot(xs, h1[3, :], label=L"h1[3]", xlabel=L"xs", ylabel=L"h1_{3}", title=L"h_{13}=a(β_{03}+Ω_{03})")
	plot(plot4, plot5, plot6, ylimit=(0, 1), lw=2, titlefontsize=12)
end # let

# ╔═╡ 85353649-2a95-45fd-9720-5270189df673
begin 
	plot7 = 
		plot(xs, ysHat', label=L"\hat ys", xlabel=L"xs", ylabel=L"\hat ys", title="Output of Generator (= Shallow 1-3-1 Network)", titlefontsize=11, lw=2, color=:red)
	plot(plot7)
end # begin

# ╔═╡ 641d1183-318a-4e3c-8392-cae892b218ea
md"
---
##### 3. Shallow 1-3-1 LUX-Model
###### 3.1 Training Loop
"

# ╔═╡ 28083194-281d-49ee-84b0-cec56e3a039f
myCorSq = zeros(Float64, 20)                #  = proportions of explained variance

# ╔═╡ c06c7d20-324d-41fb-865e-111ebf627cc4
function trainTheModel(luxModel, xsData::Vector{Float32}, ysData::Vector{Float32}; title="LUX-Model")
	#-------------------------------------------------------------------------------
	function myMse(model, parameters, states, data)
		# dereferencing of 'data'-tuple in two elements 'xs' and 'ys'
		xs = data[1]          # 1st element of tuple data
		ys = data[2]          # 2nd element of tuple data
		# ysPredicted, states = Lux.apply(model, data[1], parameters, states)
		# Lux.apply(model, data[1], parameters, states) == model(xs, parameters, states)
		ysPredicted, states = model(xs, parameters, states)
	    mseLoss = mean(abs2, ysPredicted .- ys)
	    mseLoss, states, ()
	end
	#-------------------------------------------------------------------------------
	function iterationLoop(trainState, vjp, data)
		lossOld = Inf
		lossNew = 0.0
		epochs  = 0
		#---------------------------------------------------------------------------
		while !isapprox(abs(lossOld - lossNew), 0.0)
			lossOld = lossNew
			epochs += 1 
			grads, lossNew, stats, trainState = 
				Lux.Training.compute_gradients(vjp, myMse, data, trainState)
		    if (epochs % 10) == 0  
				println(lazy"MSE-Loss Value after $epochs epochs: $lossNew")
			end # if
			trainState = Lux.Training.apply_gradients(trainState, grads)
		end # while
		#---------------------------------------------------------------------------
		println(lazy"MSE-Loss Value after $epochs epochs: $lossNew")    # final ssq result
		ssqLoss = lossNew * N
		println(lazy"SSQ-Loss Value after $epochs epochs: $ssqLoss")    # final mse result)
		trainState, lossNew, epochs   
	end # function iterationLoop
	#-------------------------------------------------------------------------------
	learningRate = 0.03f0 
	# Optimizer: ADAM (ADAptive Moment Estimation)
	optimizer = Adam(learningRate)
	#-------------------------------------------------------------------------------
	rng = MersenneTwister()
	Random.seed!(rng, 12345)
	# Lux.Experimental.TrainState is a wrapper over parameters, states 
	#   and optimizer states.
	trainState = Lux.Experimental.TrainState(rng, luxModel, optimizer)
	#-------------------------------------------------------------------------------
	# struct ADTypes.AutoZygote
	vjp_rule = AutoZygote() 
	dev_cpu  = cpu_device()
	# dev_gpu  = gpu_device()
	#-------------------------------------------------------------------------------
	trainStateNew, mseLoss, epochs =  
		#  'data' gets packaged as a tuple. 
		#  This is dereference in myMse-loss function
		iterationLoop(trainState, vjp_rule, (xsData', ysData'))
	#-------------------------------------------------------------------------------
	ysPredNew = 
		trainStateNew.model(xsData', trainStateNew.parameters, trainStateNew.states)[1]
	#-------------------------------------------------------------------------------
	myCor = cor(ysData, ysPredNew')[1]
	myCorSquare = myCor^2
	pushfirst!(myCorSq, myCorSquare)
	#-------------------------------------------------------------------------------
	Plots.scatter(xsData, ysData, label=L"data:(x, y)", titlefontsize=9, title=title)
	Plots.plot!(xsData, ysPredNew', label=L"model: \hat ys", color=:red, lw=3)
	Plots.plot!(xs, ysHat[1, :], label=L"generator: \hat ys", ls=:dash, lw=2.0, color=:red)
	annotate!(0.40, -0.28, "mse = $mseLoss", 8)
	annotate!(0.40, -0.34, "r(y, y-hat) = $myCor", 8)
	annotate!(0.40, -0.40, "r(y, y-hat)^2 = $myCorSquare", 8)
	annotate!(0.40, -0.46, "#epochs = $epochs", 8)
	#-------------------------------------------------------------------------------
end # function trainTheModel

# ╔═╡ ce17b890-56aa-4312-8210-60eb73bd0ba8
md"
---
###### 3.2 relu-Model Specification
"

# ╔═╡ 067ac55e-5624-44d2-b878-db502cc0f9f5
shallow_1_3_1_relu_Model = 
	Chain(
		Dense(1 => 3, relu), 
		Dense(3 => 1, relu))

# ╔═╡ 65807101-61b9-4f46-a986-eda7b7830fd3
md"
###### 4.3 Training of relu-Model
"

# ╔═╡ 78b953b1-3ab3-42c7-8bb1-c0a2b16ad4e3
trainTheModel(shallow_1_3_1_relu_Model, xs, ysHat[1, :], title="LUX-Model with *relu*-Activation")

# ╔═╡ 4a8cde93-e991-4ecd-a6c7-dcf1c2ec20a1
md"
---
###### 3.4 tanh-Model Specification
"

# ╔═╡ 3f88f4e9-6418-4571-89fc-407505892ddb
shallow_1_3_1_tanh_Model1 = 
	Chain(
		Dense(1 => 3, tanh), 
		Dense(3 => 1, tanh))

# ╔═╡ 9f064895-8e29-4c95-a5f0-f37d7b1e8ea6
md"
---
###### 3.5 Training of tanh-Model
"

# ╔═╡ 4e0bfee4-f8a1-45e0-ba30-4a9cd4774f71
trainTheModel(shallow_1_3_1_tanh_Model1, xs, ysHat[1, :], title="LUX-Model with *tanh*-Activation")

# ╔═╡ ff142dba-7382-4047-a8e0-57fb7f447871
md"
---
##### 4. From Shallow 1-10-1 Neural Networks to [Univariate Regression](https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap05/5_1_Least_Squares_Loss.ipynb)
###### 4.1 Training Data: Single Input, Single Output
Data $ysTrain$ are generator's output $\hat ys$ with gaussian random noise $randn(Float32)/4.0$ added.
"

# ╔═╡ 06da2b96-0861-488a-9e99-baa88a044f59
xsTrain = xs

# ╔═╡ c3071b6b-7d31-4530-8df7-f56cb440c6cd
ysTrain = convert(Vector{Float32}, [ysHat[1, j] + randn(Float32)/4.0  for j in 1: length(ysHat[1, :])])

# ╔═╡ 3dca892a-c4f1-40f7-872e-04f9c494b10e
scatter(xsTrain, ysTrain, label=L"data: (x, y)", xlabel=L"xs", ylabel=L"ys", title="Training Data", titlefontsize=12)

# ╔═╡ 254842e5-96c1-431b-ae7c-d3922db15233
md"
---
###### 4.2 Generator's Output
The figure below is similar but not identical to one in Prince's notebook 5.1. Here we used the shallow 1-3-1 generator to first generate the error-free $\hat ys$ and then the noisy $ysTrain$.
"

# ╔═╡ 005ccb7b-8048-44c6-a66b-d067cd4592ae
let stDevResid = std(ysTrain - ysHat[1, :])  # standard deviation of residuals
	#-----------------------------------------------------------------------------
	scatter(xsTrain, ysTrain, label=L"data: (xsTrain, ysTrain)", title="Training Data", titlefontsize=12)
	#-----------------------------------------------------------------------------
	plot!(xs, ysHat[1, :], label=L"generator: \hat ys", xlabel=L"xs", ylabel=L"ys", title="Output of Shallow 1-3-1 Generator and Training Data", titlefontsize=11, lw=2.0, color=:red)
	#-----------------------------------------------------------------------------
	plot!(xs, ysHat[1, :] .+ (1.96 * stDevResid), label=L"generator: \hat ys+1.96s", xlabel=L"xs", ylabel=L"ys", ls=:dash, color=:black)
	#-----------------------------------------------------------------------------
	plot!(xs, ysHat[1, :] .- (1.96 * stDevResid), label=L"generator: \hat ys-1.96s", xlabel=L"xs", ylabel=L"ys", ls=:dash, color=:black)
	#-----------------------------------------------------------------------------
end # let

# ╔═╡ 6ebbc969-247d-4434-9ccc-d79a054420f6
md"
---
###### 4.3 LUX.jl Shallow Models
"

# ╔═╡ 7e1a7052-8d8f-4ee1-aa3d-5e89392d0625
md"
###### 4.3.1 Shallow 1-10-1 Model with tanh-Activation
"

# ╔═╡ 4f6dc297-5450-4475-9922-4008d893c1a0
shallow_1_10_1_tanh_Model = 
	Chain(Dense(1 => 10, tanh), Dense(10 => 1, tanh))

# ╔═╡ c43b7f93-fcc0-4d37-a34c-af585eececae
trainTheModel(shallow_1_10_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_10_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 1d721540-eaa9-44be-99b3-cb582d1a7a27
md"
---
###### 4.3.2 Shallow 1-9-1 Model with tanh-Activation
"

# ╔═╡ b11279a0-28e2-4e37-a2a8-92c1837a31a2
shallow_1_9_1_tanh_Model = 
	Chain(Dense(1 => 9, tanh), Dense(9 => 1, tanh))

# ╔═╡ cf4658ff-18b0-4578-a78c-070128826da7
trainTheModel(shallow_1_9_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_9_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 983567af-c203-4590-aaab-702549cc36c9
md"
---
###### 4.3.3 Shallow 1-8-1 Model with tanh-Activation
"

# ╔═╡ 5f9fed69-4a9f-4673-a04a-fafee40c6f9f
shallow_1_8_1_tanh_Model = 
	Chain(Dense(1 => 8, tanh), Dense(8 => 1, tanh))

# ╔═╡ e3f07344-3757-4f8b-b230-18249da7925e
trainTheModel(shallow_1_8_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_8_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 82b35ffc-3e85-4770-a5af-e276f52d9488
md"
---
###### 4.3.4 Shallow 1-7-1 Model with tanh-Activation
"

# ╔═╡ cb3aac6c-108a-40c5-b33f-5718cb306779
shallow_1_7_1_tanh_Model = 
	Chain(Dense(1 => 7, tanh), Dense(7 => 1, tanh))

# ╔═╡ 2c00a7c4-4b7b-45a2-892d-c90239fd6b4a
trainTheModel(shallow_1_7_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_7_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 3ecc19dc-2f88-44b6-9f32-0ecc220bf268
md"
---
###### 4.3.5 Shallow 1-6-1 Model with tanh-Activation
"

# ╔═╡ 8075984e-3a8d-4582-b07d-86f0baa17d0f
shallow_1_6_1_tanh_Model = 
	Chain(Dense(1 => 6, tanh), Dense(6 => 1, tanh))

# ╔═╡ 940da618-0c3a-4a7e-92e4-304eeafeef16
trainTheModel(shallow_1_6_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_6_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 0ea44381-9c3c-4a99-9dc4-5fa2ee36cb5f
md"
###### 4.3.6 Shallow 1-5-1 Model with tanh-Activation
"

# ╔═╡ 86f7a201-c5d0-4369-afe7-2ba5227c4be1
shallow_1_5_1_tanh_Model = 
	Chain(Dense(1 => 5, tanh), Dense(5 => 1, tanh))

# ╔═╡ 449ff9af-4bca-4969-9183-d606bd404a5e
trainTheModel(shallow_1_5_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_5_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 3dd7730f-292a-480a-9f8b-2cf4aedfe14c
md"
---
###### 4.3.7 Shallow 1-4-1 Model with tanh-Activation
"

# ╔═╡ c962252b-2161-4067-a358-401d825c3fba
shallow_1_4_1_tanh_Model = 
	Chain(Dense(1 => 4, tanh), Dense(4 => 1, tanh))

# ╔═╡ 1c3cf9da-0836-4f45-9ee9-dcede8afb43e
trainTheModel(shallow_1_4_1_tanh_Model, xsTrain, ysTrain, 
	title="LUX.jl shallow_1_4_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 40266303-1552-4aa3-91b6-aaa940e72e50
md"
---
###### 4.3.8 Shallow 1-3-1 Model with tanh-Activation
"

# ╔═╡ 318b1feb-8ec8-4212-98d2-4852aaf7d0d6
shallow_1_3_1_tanh_Model2 = 
	Chain(Dense(1 => 3, tanh), Dense(3 => 1, tanh))

# ╔═╡ c2653965-7419-43bd-b77c-33c564a68722
trainTheModel(shallow_1_3_1_tanh_Model2, xsTrain, ysTrain, title="LUX.jl shallow_1_3_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 86a74fe4-b431-4251-b16c-211c0838503a
typeof(xsTrain), typeof(ysTrain)

# ╔═╡ 91095d13-7350-4333-839c-2e80445b5eee
md"
---
###### 4.3.9 Shallow 1-2-1 Model with tanh-Activation
"

# ╔═╡ dc34ffc6-2a3f-4fe0-b4f9-1f02ecc41acc
shallow_1_2_1_tanh_Model = 
	Chain(
		Dense(1 => 2, tanh), Dense(2 => 1, tanh))

# ╔═╡ 8cba4700-315a-4fb8-a185-9ab9c7aaee44
trainTheModel(shallow_1_2_1_tanh_Model, xsTrain, ysTrain, title="LUX.jl shallow_1_2_1_tanh_Model with *tanh*-Activation")

# ╔═╡ 0af08cd3-f624-4a2b-938d-8bde2d362812
md"
---
###### 4.3.10 Shallow 1-1-1 Model with tanh-Activation
"

# ╔═╡ 2cba1418-0a5d-45ee-90ea-c0d05cb26cfd
shallow_1_1_1_tanh_Model = 
	Chain(
		Dense(1 => 1, tanh),
		Dense(1 => 1, tanh))

# ╔═╡ dee8d6ee-cf5b-4447-bdfb-2b5c8e3b37e5
trainTheModel(shallow_1_1_1_tanh_Model, xsTrain, ysTrain, title="LUX.jl shallow_1_1_1_tanh_Model with *tanh*-Activation")

# ╔═╡ c127cc4a-289b-49e9-a4e4-4a21b52c2ee4
md"
---
###### 4.3.11 Univariate 1-1 Regression Model with tanh-Activation
"

# ╔═╡ 3ca02ecf-7d83-43c0-b9d3-57c1a3ac758e
univariate_1_1_tanh_RegressionModel = Dense(1 => 1, tanh)

# ╔═╡ ed7f76ea-57d3-4f84-bdca-8c562d25ad2b
trainTheModel(univariate_1_1_tanh_RegressionModel, xsTrain, ysTrain, title="LUX.jl Univariate_1_1_tanh_Regression_Model with *tanh*-Activation")

# ╔═╡ a36eed97-8bce-4aff-8935-ec31b7a6b930
myCorSq

# ╔═╡ a0dfcbd1-3d6b-49a2-9f13-60be76387973
begin
	scatter(1:1:11, myCorSq[11:-1:1], ylimit=(0, 1), label=L"(model, r^2)", xlabel="Model", ylabel=L"r^2", title="Drop in Proportion of Variance Explanations", titlefontsize=12)
	plot!(1:1:11, myCorSq[11:-1:1], label=L"(model, r^2)", lw=2.5)
end # begin

# ╔═╡ 1217056a-dc3d-4acd-92ce-f86aca4aa99e
md"
---
##### 5. Summary
We reimplemented Prince's $10$-parameter *shallow 1-3-1 relu-model* and used it as a generator for error-free data $\hat ys$. To that we added gaussian distributed noise to obtain training data $ysTrain$. Training data $xsTrain:=xs, ysTrain$ were fed into various LUX.jl models with a range of $10$ to $0$ hidden units. The last model is better known as *univariate line regression*. 

For each model we computed parameters minimzing *mean squares loss* and the correlation $r$ and its square $r^2$ between model predictions $\hat ys$ and $ysTrain$. It could be demonstrated that we could reduce model complexity down to the
$shallow\_1\_3\_1\_tanh\_Model$ with $10$ parameters without substantially deteriorating the proportion of explained variance $r^2$.

The remarkable result is, that our *optimal* LUX.jl $shallow\_1\_3\_1\_tanh\_Model$ possesses the same structure as the original latent generator despite the fact that both models use different activations.
"

# ╔═╡ 2048da89-ff72-4a4e-b8ff-696acb9e9555
md"
---
##### References

- **Bishop, C.M. & Bishop, H.**; *Deep Learning: Foundations and Concepts*, Cham, Swiss: Springer, 2024

- **Held, L. & Bové, D.S.**; *Likelihood and Bayesian Inference*, Berlin: Springer, 2nd/e, 2020

- **Prince, S.J.D.**; [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/); MIT Press, 2024, last visit 2024/07/05

- **Prince, S.J.D.**; [*Notebook 5.1 – Least Squares Loss*](https://udlbook.github.io/udlbook/); last visit 2024/07/05

- **Wikipedia**; [*Probability density function*](https://en.wikipedia.org/wiki/Probability_density_function); last visit 2024/07/08

- **Wikipedia**; [*Probability mass function*](https://en.wikipedia.org/wiki/Probability_mass_function); last visit 2024/07/08

"


# ╔═╡ 9498ad5f-d5d0-4a09-9797-fa1c284a5e4d
md"
====================================================================================

This is a **draft** under the Attribution-NonCommercial-ShareAlike 4.0 International **(CC BY-NC-SA 4.0)** license. Comments, improvement and issue reports are welcome: **claus.moebus(@)uol.de**

===================================================================================
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ADTypes = "~0.2.6"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
Lux = "~0.5.17"
Optimisers = "~0.3.2"
Plots = "~1.40.1"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "5c85f85ebe16bf108e788b8597cb9db036a62bdd"

[[deps.ADTypes]]
git-tree-sha1 = "41c37aa88889c171f1300ceac1313c06e891d245"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.6"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c5aeb516a84459e0318a02507d2261edad97eb75"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "4e42872be98fa3343c4f8458cbda8c5c6a6fa97c"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.63.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "ad25e7d21ce10e01de973cdc68ad0f850a953c52"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.21.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "9c4708e3ed2b799e6124b5673a712dda0b596a9b"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "166c544477f97bbadc7179ede1c1868e0e9b426b"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.7"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "47e4686ec18a9620850bad110b79966132f14283"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.0.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3458564589be207fa6a77dbbf8b97674c9836aab"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "77f81da2964cc9fa7c0127f941e8bce37f7f1d70"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.2+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ac7b73d562b8f4287c3b67b4c66a5395a19c1ae8"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "5d8c5713f38f7bc029e26627b687710ba406d0dd"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.12"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4e0cb2f5aad44dcfdc91088e85dee4ecb22c791c"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.16"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "9e70165cca7459d25406367f0c55e517a9a7bfe7"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.5.0"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "114e3a48f13d4c18ddd7fd6a00107b4b96f60f9c"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.28+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "ChainRulesCore", "ConcreteStructs", "ConstructionBase", "Functors", "GPUArraysCore", "LinearAlgebra", "LuxCore", "LuxDeviceUtils", "LuxLib", "MacroTools", "Markdown", "Optimisers", "PrecompileTools", "Random", "Reexport", "Setfield", "SparseArrays", "Statistics", "TruncatedStacktraces", "WeightInitializers"]
git-tree-sha1 = "d6bb9178ad7370a49121930e635840ddc383e144"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "0.5.17"

    [deps.Lux.extensions]
    LuxChainRulesExt = "ChainRules"
    LuxComponentArraysExt = "ComponentArrays"
    LuxComponentArraysReverseDiffExt = ["ComponentArrays", "ReverseDiff"]
    LuxFluxTransformExt = "Flux"
    LuxLuxAMDGPUExt = "LuxAMDGPU"
    LuxReverseDiffExt = "ReverseDiff"
    LuxTrackerExt = "Tracker"
    LuxZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    LuxAMDGPU = "83120cb1-ca15-4f04-bf3b-6967d2e6b60b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCore]]
deps = ["Functors", "Random", "Setfield"]
git-tree-sha1 = "18fcc5e8334a6c3a9c5aa2a9a1a7aef7a887c4ad"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "0.1.8"

[[deps.LuxDeviceUtils]]
deps = ["Adapt", "ChainRulesCore", "Functors", "LuxCore", "PrecompileTools", "Preferences", "Random", "SparseArrays"]
git-tree-sha1 = "aa5488d6e00397db550a220a82cf4746a9f223ac"
uuid = "34f89e08-e1d5-43b4-8944-0b49ac560553"
version = "0.1.14"

    [deps.LuxDeviceUtils.extensions]
    LuxDeviceUtilsFillArraysExt = "FillArrays"
    LuxDeviceUtilsGPUArraysExt = "GPUArrays"
    LuxDeviceUtilsLuxAMDGPUExt = "LuxAMDGPU"
    LuxDeviceUtilsLuxCUDAExt = "LuxCUDA"
    LuxDeviceUtilsMetalGPUArraysExt = ["GPUArrays", "Metal"]
    LuxDeviceUtilsRecursiveArrayToolsExt = "RecursiveArrayTools"
    LuxDeviceUtilsZygoteExt = "Zygote"

    [deps.LuxDeviceUtils.weakdeps]
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    LuxAMDGPU = "83120cb1-ca15-4f04-bf3b-6967d2e6b60b"
    LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxLib]]
deps = ["ChainRulesCore", "KernelAbstractions", "Markdown", "NNlib", "PrecompileTools", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "186650b8b54b80607491b626a8b4a27ff179942c"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "0.3.10"

    [deps.LuxLib.extensions]
    LuxLibForwardDiffExt = "ForwardDiff"
    LuxLibLuxCUDAExt = "LuxCUDA"
    LuxLibLuxCUDATrackerExt = ["LuxCUDA", "Tracker"]
    LuxLibReverseDiffExt = "ReverseDiff"
    LuxLibTrackerExt = "Tracker"

    [deps.LuxLib.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "877f15c331337d54cf24c797d5bcb2e48ce21221"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.12"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "264b061c1903bc0fe9be77cb9050ebacff66bb63"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PartialFunctions]]
deps = ["MacroTools"]
git-tree-sha1 = "47b49a4dbc23b76682205c646252c0f9e1eb75af"
uuid = "570af359-4316-4cb7-8c74-252c00c2016b"
version = "1.2.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "c4fa93d7d66acad8f6f4ff439576da9d2e890ee0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.1"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "7b0e9c14c624e435076d19aea1e5cbdec2b9ca37"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.2"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "1b0b1205a56dc288b71b1961d48e351520702e24"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.17"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WeightInitializers]]
deps = ["ChainRulesCore", "PartialFunctions", "PrecompileTools", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d8d2cc5d798da563677cadea14d4e69af7f6df83"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "0.1.5"

    [deps.WeightInitializers.extensions]
    WeightInitializersCUDAExt = "CUDA"

    [deps.WeightInitializers.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4ddb4470e47b0094c93055a3bcae799165cc68f1"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.69"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "873b4f805771d3e4bafe63af759a26ea8ca84d14"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.42+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─08e3a970-3ac9-11ef-30de-ad60816a0a04
# ╟─73d1e53f-1090-42c0-b76d-28242f907ffd
# ╟─be780bbe-645d-4342-8020-4a33704dd6a4
# ╠═64c51f97-6b98-4253-8a89-3cb22c9c80a5
# ╟─5e4dd047-f1fb-4b3a-8d7c-e320b9821116
# ╟─67260b0d-7276-448b-8d5e-fbf8c8311a42
# ╠═deab1f59-b1a6-4d76-bbd6-97910cc04419
# ╟─b7aaf6e9-3563-47f6-8983-f59479afe703
# ╠═d792444b-69a0-4225-95b5-c46c83ef0e89
# ╠═c9c1d3c6-8fd7-4d7a-a173-5016b0bdb5cf
# ╟─831555f2-5bdb-43cd-895a-c3a5eadb3be3
# ╠═9a3bbc35-3dad-4d92-acc2-1e8dac451e81
# ╠═78258a9c-8594-437e-b0b4-cc3c29f946aa
# ╠═dd7122c4-2fba-46af-b560-f0649d4c95b5
# ╟─3f78e783-d6b9-450f-9914-e21fce295a14
# ╠═96a59f0e-d22e-4476-b5f2-69018a9a9fb8
# ╠═d7643abe-a1ff-496c-86c9-e9364cb2bd92
# ╠═1adf5c3e-4e88-4175-9cf4-78916d3eade8
# ╠═fb2441a4-59eb-4439-9f60-3baafe33699d
# ╠═73bb7782-8d52-4905-9c97-b7cf0fc44fe4
# ╠═85353649-2a95-45fd-9720-5270189df673
# ╟─641d1183-318a-4e3c-8392-cae892b218ea
# ╠═28083194-281d-49ee-84b0-cec56e3a039f
# ╠═c06c7d20-324d-41fb-865e-111ebf627cc4
# ╟─ce17b890-56aa-4312-8210-60eb73bd0ba8
# ╠═067ac55e-5624-44d2-b878-db502cc0f9f5
# ╟─65807101-61b9-4f46-a986-eda7b7830fd3
# ╠═78b953b1-3ab3-42c7-8bb1-c0a2b16ad4e3
# ╟─4a8cde93-e991-4ecd-a6c7-dcf1c2ec20a1
# ╠═3f88f4e9-6418-4571-89fc-407505892ddb
# ╟─9f064895-8e29-4c95-a5f0-f37d7b1e8ea6
# ╠═4e0bfee4-f8a1-45e0-ba30-4a9cd4774f71
# ╟─ff142dba-7382-4047-a8e0-57fb7f447871
# ╠═06da2b96-0861-488a-9e99-baa88a044f59
# ╠═c3071b6b-7d31-4530-8df7-f56cb440c6cd
# ╠═3dca892a-c4f1-40f7-872e-04f9c494b10e
# ╟─254842e5-96c1-431b-ae7c-d3922db15233
# ╠═005ccb7b-8048-44c6-a66b-d067cd4592ae
# ╟─6ebbc969-247d-4434-9ccc-d79a054420f6
# ╟─7e1a7052-8d8f-4ee1-aa3d-5e89392d0625
# ╠═4f6dc297-5450-4475-9922-4008d893c1a0
# ╠═c43b7f93-fcc0-4d37-a34c-af585eececae
# ╟─1d721540-eaa9-44be-99b3-cb582d1a7a27
# ╠═b11279a0-28e2-4e37-a2a8-92c1837a31a2
# ╠═cf4658ff-18b0-4578-a78c-070128826da7
# ╟─983567af-c203-4590-aaab-702549cc36c9
# ╠═5f9fed69-4a9f-4673-a04a-fafee40c6f9f
# ╠═e3f07344-3757-4f8b-b230-18249da7925e
# ╟─82b35ffc-3e85-4770-a5af-e276f52d9488
# ╠═cb3aac6c-108a-40c5-b33f-5718cb306779
# ╠═2c00a7c4-4b7b-45a2-892d-c90239fd6b4a
# ╟─3ecc19dc-2f88-44b6-9f32-0ecc220bf268
# ╠═8075984e-3a8d-4582-b07d-86f0baa17d0f
# ╠═940da618-0c3a-4a7e-92e4-304eeafeef16
# ╟─0ea44381-9c3c-4a99-9dc4-5fa2ee36cb5f
# ╠═86f7a201-c5d0-4369-afe7-2ba5227c4be1
# ╠═449ff9af-4bca-4969-9183-d606bd404a5e
# ╟─3dd7730f-292a-480a-9f8b-2cf4aedfe14c
# ╠═c962252b-2161-4067-a358-401d825c3fba
# ╠═1c3cf9da-0836-4f45-9ee9-dcede8afb43e
# ╟─40266303-1552-4aa3-91b6-aaa940e72e50
# ╠═318b1feb-8ec8-4212-98d2-4852aaf7d0d6
# ╠═c2653965-7419-43bd-b77c-33c564a68722
# ╠═86a74fe4-b431-4251-b16c-211c0838503a
# ╟─91095d13-7350-4333-839c-2e80445b5eee
# ╠═dc34ffc6-2a3f-4fe0-b4f9-1f02ecc41acc
# ╠═8cba4700-315a-4fb8-a185-9ab9c7aaee44
# ╟─0af08cd3-f624-4a2b-938d-8bde2d362812
# ╠═2cba1418-0a5d-45ee-90ea-c0d05cb26cfd
# ╠═dee8d6ee-cf5b-4447-bdfb-2b5c8e3b37e5
# ╟─c127cc4a-289b-49e9-a4e4-4a21b52c2ee4
# ╠═3ca02ecf-7d83-43c0-b9d3-57c1a3ac758e
# ╠═ed7f76ea-57d3-4f84-bdca-8c562d25ad2b
# ╠═a36eed97-8bce-4aff-8935-ec31b7a6b930
# ╠═a0dfcbd1-3d6b-49a2-9f13-60be76387973
# ╟─1217056a-dc3d-4acd-92ce-f86aca4aa99e
# ╟─2048da89-ff72-4a4e-b8ff-696acb9e9555
# ╟─9498ad5f-d5d0-4a09-9797-fa1c284a5e4d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002