### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 007be6ce-ca18-4877-82e4-f49f85fd993b
begin 
	using Plots
	using Distributions
	using Printf
	using LaTeXStrings, Latexify
	using Optim
end # begin

# ╔═╡ 400e7456-8a3b-44cc-ac77-5b28087b81fb
md"
=====================================================================================
#### UDL\_20240711\_5\_2\_Maximum\_Likelihood\_Estimation.jl
##### file: UDL\_20240711\_5\_2\_Maximum\_Likelihood\_Estimation\_I.jl
##### code: Julia 1.10.4/Pluto by *** PCM 2024/07/28 ***

=====================================================================================
"

# ╔═╡ e2666b94-6c11-4de3-9b68-ed61a01aa334
md"
---
##### 0. Introduction

Here we deal with various *MLE*-related concepts in the context of the Gaussian distributed random samples: density, likelihood, density and likelhood function, log-likelihood, likelihood kernel, minimization and maximization, first and second derivatives of log-likelihood function, Fisher's score function, Hessian, Fisher information, Nelder-Mead-, L-BFGS-, and Newton-algorithm.

Besides several others we use function of the Optim.jl package.
"

# ╔═╡ 19818803-5892-46a4-b0a7-2025cd0555ba
md"
##### 1. Packages
"

# ╔═╡ 3a2a4fca-9e51-4882-967a-4d62718dae94
md"
---
##### 2. Maximum Likelihood Function

###### 2.1 Model Function of Statistical Model

$\theta = h(x, \mathbb \phi)$

where:

$h = \text{input-output function of (e.g. regression or neural-net) model}$

$\;$

$x = \text{ input datum (e.g. realization)}\ x \text{ of independent random variable } X:\ X=x$

$\;$

$\mathbb \phi = \text{ vector of (unknown) model parameters (eg. regression or neural net coefficients)}$

$\;$
$\;$

"

# ╔═╡ 2d21c9e7-afb3-4354-8bb0-36dd672b3dcb
md"
---
###### 2.2 Probability Density Function (PDF) and Probability Mass Function (PMF)

###### 2.2.1 Definition
Definitions of the *continuous density function (PDF)* can be found [here](https://en.wikipedia.org/wiki/Probability_density_function) and that of the *discrete probability (density) function (PMF)* can be found [here](https://en.wikipedia.org/wiki/Probability_mass_function). Both can be denoted by the symbol $f$ if its clear from the context which semantics is meant. In the context of statistical models the model function $h(x, \phi)$ predicts the classical parameter $\theta$ in $f(x; \theta)$ so that we replace $\theta$ by the model function $h(x, \phi)$:

$f(y; h(x, \phi), x, \phi) = \begin{cases} \text{ PDF } & \text{ if defined on a real random variable } X \\
\text{ PMF } & \text{ if defined on a discrete random variable } X \\ 
\end{cases}$

$\;$
$\;$

"

# ╔═╡ 85caf5f0-d836-4cb8-a078-0894ae07016d
md"
###### 2.2.2 Example: PDF of [Gaussian Distributions](https://en.wikipedia.org/wiki/Normal_distribution) (with Model Function as Parameter Substitute)

$f_{\mathcal N}(y; h(\phi)) = \mathcal N(y; h(\phi)) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp\left(-\frac{1}{2}\left(\frac{y-\mu}{\sigma}\right)^2\right)$

$\;$
$\;$

where: 

$\mathbf \phi = (\mu, \sigma)$ 

$\;$

and here:

$h(\phi) := I(\phi) = \phi \text{; the identity function }$

$\;$
$\;$
"

# ╔═╡ f56ef177-68fc-4cb0-94aa-4b8254b62406
function myGaussianPDF(x; μ=0.0, σ=1.0)
	(1/(√(2.0*π*σ^2)))*exp(-((1.0/2.0)*((x-μ)/σ)^2))
end # function myGaussianPDF

# ╔═╡ e116bbe9-bca6-4f7b-94cb-fdafa52f6ddd
myGaussianPDF(0.0)

# ╔═╡ 7c4bd2ef-b479-4708-ac63-90ee118bea47
begin # exercise in Prince's notebook 5.2 with own code
	@printf("myGaussianPDF(1.0, μ=-1.0, σ=2.3) = %4.3f", 
		myGaussianPDF(1.0, μ=-1.0, σ=2.3))
	myGaussianPDF(1.0, μ=-1.0, σ=2.3)
end # begin

# ╔═╡ f94fc7fb-9f21-4c99-a0f4-ddf4a4df37c9
begin # exercise in Prince's notebook 5.2 with library functions pdf, Normal
	@printf("pdf.(Normal(-1.0, 2.3), 1.0) = %4.3f", 
		pdf.(Normal(-1.0, 2.3), 1.0))
	pdf.(Normal(-1.0, 2.3), 1.0)
end # begin

# ╔═╡ 95af722f-9675-4562-a115-166bb5e27679
let zs = [z for z in -5.0:0.1:+5.0]
	myGaussianPDFs = [myGaussianPDF(z) for z in zs] 
	plot(zs, myGaussianPDFs, label=L"(z, N(z;\mu=0.0, \sigma=1.0))", xlabel=L"z", ylabel=L"N(z;\mu=0, \sigma=1)", title="Density of Standard Gaussian", titlefontsize=12)
	plot!([0.0, 0.0], [0.0, myGaussianPDF(0.0)], ls=:dash, label=L"(0, N(0;\mu=0, \sigma=1))")
	plot!([-5.0, 5.0],[0.0, 0.0], color=:black, label=L"(z, 0.0)")
end # let

# ╔═╡ 25584038-b162-438c-acc7-597fb5b65b0d
let ys = [y for y in -7.0:0.1:+5.0]
	myGaussianPDFs = [pdf.(Normal(-1.0, 2.3), y) for y in ys] 
	plot(ys, myGaussianPDFs, label=L"(y, N(y;\mu=-1, \sigma=2.3))", xlabel=L"y", ylabel=L"N(y;\mu=0.0, \sigma=1.0)", title="Density of Distributions.Normal(-1.0, 2.3)", titlefontsize=12)
	plot!([1.0, 1.0],[0.0, pdf.(Normal(-1.0, 2.3), 1.0)], ls=:dash, label=L"(1, N(1;\mu=-1, \sigma=2.3)")
	plot!([-7.0, 5.0],[0.0, 0.0], color=:black, label=L"(z, 0.0)")
end # let

# ╔═╡ 479adfc6-00da-4279-b64d-9f8ef68d7619
md"
---
###### 2.2.3 Small Sample of Gaussian Distributed Random Variable
"

# ╔═╡ 303aaffa-09aa-4c80-a92d-90045a7b688f
begin # Brown, 2014, p.70, Fig 3.1
	ys = [75, 87, 89, 90, 93, 99, 100, 101, 103, 112, 116, 135] 
	n = length(ys)
	ybar = mean(ys)
	ystddev  = std(ys)
	n, ybar, ystddev
end # begin

# ╔═╡ 127347fa-1ac3-4306-acc9-61091538ad71
let norm_100_15PDF = Normal(100, 15)
	ysPDFs = [pdf(norm_100_15PDF, y) for y in ys]  # densities N(y; μ=100, σ=15)
	for l in 1:length(ys)
		@printf "density(%3i) = %5.4F \n" ys[l] ysPDFs[l]
	end # for
	scatter(ys, ysPDFs, label=L"density: (x,N(x;100,15))", xlabel=L"x", ylabel=L"N(y; 100, 15)", title="Sample of Density Heights (cf. Brown, 2014, p.70, Fig. 3.1)", titlefontsize=12)
	plot!(ys, ysPDFs, label=L"density: (y,N(y;100,15))", lw=2)
end # let

# ╔═╡ 23e54ade-e9b0-47ae-91d4-fde9b5e3ec67
md"
---

###### 2.3 Likelihood Function and Likelihood Kernel
###### 2.3.1 Definition
The *likelihood function* $\mathcal L$ is an indicator of the *plausbility* of various trial or guess values $x$ for the *fixed, unknown* parameter $\phi$. In comparison to the *density* function $f$ the *conceptual* roles of $x$ and $\phi$ are now *reversed* (Held & Bove, 2020, p.14). Now, data are *fixed* (after their observation) and $\mathbb \phi$ is *hypthetical* and *mutable*. This is due to the *counterfactual reasoning* process: we are looking for that value of $\phi$ for which the *plausability* of the data is maximized.

$\mathcal L(\phi; f, h, x, y) = f(y;h(x,\phi), x, \phi)$

$\;$

The likelihood of $n$ realizations $x_{1:n}$ of a random variable $X$ is the product of their single densities. This is true under the assumption of *i.i.d. (= independent identical distributed)* random realizations $X=x_{1:n}$.

$\mathcal L(\phi; f, h, x_{1:n}, y_{1:n}) = \prod_{i=1}^n f\left(y_i; h(x_i, \mathbb \phi), x_i, \phi \right)$

$\;$
$\;$
$\;$

To simplify computations sometimes the likelihood function can be stripped off multiplicative constants. The purged result is called *likelihood kernel* (Held & Bove, 2020, p.14).

$\;$
$\;$

"

# ╔═╡ 54e75a6a-8ea9-41dc-ae12-e3691e0f893e
md"
###### 2.3.2 Example: Likelihood Function of a Set of Gaussian Distributed Realizations $Y=y_i$

$f_{\mathcal N}\left(y; h(\mu, \sigma)\right) = \mathcal N(y; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp\left(-\frac{1}{2}\left(\frac{y-\mu}{\sigma}\right)^2\right)$

$\;$
$\;$
$\;$

$\mathcal L(\mu;f_{\mathcal N}, \sigma, y_{1:n}) = \prod_{i=1}^n \mathcal N(y_i; \mu, \sigma)= \prod_{i=1}^n \left\{\frac{1}{\sqrt{2 \pi \sigma^2}}\exp\left(-\frac{1}{2}\left(\frac{y-\mu}{\sigma}\right)^2\right)\right\}=$

$\;$
$\;$
$\;$

$=\left(\frac{1}{\sqrt{2 \pi \sigma^2}}\right)^n exp\left(-\frac{1}{2}\sum_{i=1}^n \left(\frac{y_i-\mu}{\sigma}\right)^2\right) =$

$\;$
$\;$
$\;$

$= \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right).$

$\;$
$\;$
$\;$

The second term is the *likelihood kernel*.

$\;$

"

# ╔═╡ 38ac97c9-4a83-4740-8444-2fac23fe31ce
md"
###### 2.3.3 Example: Gaussian Likelihood as *Product of  Densities*.
Gaussian likelihood under the assumption of known $\sigma$: 

$\mathcal L_{\mathcal N}(μ; f_{\mathcal N}, \sigma, y_{1:n}) = \prod_{i=1}^n f_{\mathcal N}(\mu; y_i, \sigma) = \prod_{i=1}^n \left\{\frac{1}{\sqrt{2 \pi \sigma^2}}\exp\left(-\frac{1}{2}\left(\frac{y_i-\mu}{\sigma}\right)^2\right)\right\}$

$\;$
$\;$
$\;$

"

# ╔═╡ 2ea367b2-cedd-4720-a487-1dfacd1ed91a
function likelihoodGaussianProductVersion(xs::Vector, gaussianPDF::Function)
	# product of single densities or likelihoods
	prod(gaussianPDF, ys, init=1.0)  
end # function likelihoodGaussianProductVersion

# ╔═╡ bbb2c7b7-1f2a-4daf-9bc3-dff8e7eb5e22
let μs1  = [μ for μ in 98.0:1:102.0]
	μs2  = [μ for μ in 98.0:0.1:102.0]
	ls1 = [likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=μ, σ=15.0)) for μ in μs1]
	ls2 = [likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=μ, σ=15.0)) for μ in μs2]
	plot(μs2, ls2, label=L"(μ, L(\mu\;x,\sigma))", xlabel=L"\mu", ylabel=L"L(\mu;x,\sigma)", title="Product of Likelihoods (cf. Brown, 2014, Tab.3.1, p.72)", titlefontsize=12)
	scatter!(μs1, ls1, label=L"(μ, L(\mu\;y,\sigma))")
end # let

# ╔═╡ 50c9f7e8-1e09-4718-b012-aad4031bbfa0
md"
The warning message *'No strict ticks found'* is an indicator that the y-values plotted are very small numbers.

The following values of Gaussian likelihood are identical to those in Brown's Tab. 3.1 (Brown, 2014, Tab.3.1,  p.72)
"

# ╔═╡ 3b93960e-1f23-4be1-a1e9-05761d6bd263
likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=98.0, σ=15.0))

# ╔═╡ 1e0f445a-4bf1-41db-b0c8-9f31b734cf8e
likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=99.0, σ=15.0))

# ╔═╡ 4957f59d-eaff-4719-b3a2-b4b4b785411c
likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=100.0, σ=15.0))

# ╔═╡ c747204a-9ffe-4344-a5b6-214a30e95bb0
likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=101.0, σ=15.0))

# ╔═╡ 00467aab-cfbc-4ce3-a02e-620eb32251cc
likelihoodGaussianProductVersion(ys::Vector, y -> myGaussianPDF(y; μ=102.0, σ=15.0))

# ╔═╡ 6b5c88e9-4869-4820-972f-71f0988bed35
md"
###### 2.3.4 Example: Gaussian Likelihood as Weighted *Sum of Squares* in Argument of $exp$
A further simplified of the Gaussian likelihood contains a *sum of squares* expression in the exponent:


$\mathcal L(μ; f_{\mathcal N}, \sigma, y_{1:n}) = \prod_{i=1}^n f_{\mathcal N}(\mu; y_i, \sigma) = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
"

# ╔═╡ 546d9ae5-3104-4124-862c-f09f16d61806
function likelihoodOfGaussianSumVersion(xs::Vector; μ=0.0, σ=1.0)
	(1.0/√(2.0*π*σ^2))^n * exp((-1.0/(2.0*σ^2)) * sum((xs .- μ).^2))
end # function likelihoodOfGaussianSumVersion

# ╔═╡ 77f301fc-1e9e-49a4-8549-9d0918a2bd6e
let μs1  = [μ for μ in 98.0:1:102.0]
	μs2  = [μ for μ in 98.0:0.1:102.0]
	ls1 = [likelihoodOfGaussianSumVersion(ys::Vector, μ=μ, σ=15.0) for μ in μs1]
	ls2 = [likelihoodOfGaussianSumVersion(ys::Vector, μ=μ, σ=15.0) for μ in μs2]
	plot(μs2, ls2, label=L"(μ, L(\mu\;y,\sigma))", xlabel=L"\mu", ylabel=L"L(\mu;y,\sigma)", title="Sum Formula for Likelihoods (cf. Brown, 2014, Tab.3.1, p.72)", titlefontsize=10)
	scatter!(μs1, ls1, label=L"(μ, L(\mu\;y,\sigma))")
end # let

# ╔═╡ a94453a2-7724-4a47-a952-9efed4bdd197
md"
The following values of Gaussian likelihood are identical to those in Brown's Tab. 3.1 (Brown, 2014, Tab.3.1,  p.72)
"

# ╔═╡ 6d01b17d-7a2a-4689-91aa-3a941c22fdfa
likelihoodOfGaussianSumVersion(ys::Vector, μ=98.0, σ=15.0)

# ╔═╡ 79527167-096e-47f3-b67f-38724ab7bab3
likelihoodOfGaussianSumVersion(ys::Vector, μ=99.0, σ=15.0)

# ╔═╡ 67daba69-328e-4792-b6cc-65ab9de4096a
likelihoodOfGaussianSumVersion(ys::Vector, μ=100.0, σ=15.0)

# ╔═╡ 213fc124-d74e-44aa-9c6e-e5b314908c2f
likelihoodOfGaussianSumVersion(ys::Vector, μ=101.0, σ=15.0)

# ╔═╡ 97b79e30-1531-4cc9-887b-5b8549387b0f
likelihoodOfGaussianSumVersion(ys::Vector, μ=102.0, σ=15.0)

# ╔═╡ 856646cb-6fdc-4acf-bfe1-ba4c110b7710
md"
---
###### 2.4 Log-Likelihood Function

###### 2.4.1 Definition

$log\left(\mathcal L(\mathbb \phi|h, x_{1:n}, y_{1:n})\right) = \mathcal l(\mathbb \phi|h, x_{1:n}, y_{1:n})$

$\;$
$\;$

###### 2.4.2 Example: Log-Likelihood of Hypothesized $\mu$ in Gaussians with Known σ

$log\left(\mathcal L_{\mathcal N}(\mathbb \mu;\sigma, y_{1:n})\right) = \mathcal l_{\mathcal N}(\mathbb \mu;\sigma, y_{1:n})$

$\;$

$= log\left(\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)\right) =$

$\;$
$\;$
$\;$

$= n\cdot log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2 =$

$\;$
$\;$
$\;$

$= n\cdot \left(log(1) - log(\sqrt{2\pi\sigma^2})\right) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2 =$

$\;$
$\;$
$\;$

$= n\cdot \left(0 - log(\sqrt{2\pi\sigma^2})\right) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2 =$

$\;$
$\;$
$\;$

$= - n \cdot log(\sqrt{2\pi\sigma^2}) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2$

$\;$
$\;$
$\;$

The second term from above is the *log-likelihood kernel*: 

$- \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2$ 

$\;$
$\;$
$\;$

It is obvious that the log-likelihood $\mathcal l_N$ is *maximized* when the *sum-of-squares* 

$\sum_{i=1}^n (y_i-\mu)^2$

$\;$
$\;$
$\;$

in the *log-likelihood kernel* is *minimized*.

$\;$

"

# ╔═╡ 84bc6149-3c9f-467e-9383-219b6b8a9ac6
function logLikelihoodOfGaussianSumVersion(ys::Vector; μ=0.0, σ=1.0)
	-n*log(√(2.0*π*σ^2)) - (1/(2*σ^2))*sum((ys .- μ).^2)
end # function logLikelihoodOfGaussianSumVersion

# ╔═╡ dee1689b-191b-44a2-8fd4-767e2a3d4f63
md"
This log-likelihood value is identical to the value in Brown, 2014, p.73, Table 3.2
"

# ╔═╡ 229dba96-9b89-4f98-970a-62b21823d912
# cf. Brown, 2014, p.73, Table 3.2
logLikelihoodOfGaussianSumVersion(ys::Vector, μ=98.0, σ=15.0) 

# ╔═╡ 98cb5124-dfd6-456d-b72d-24fd8732cebb
# ###### log-likelihood-function using Distributions.jl
# cf. Brown, 2014, p.73, Table 3.2
sum(loglikelihood.(Normal(98.0, 15.0), ys))

# ╔═╡ aff7988f-b8e5-4207-905b-a29bc4ddae9a
let μs1  = [μ for μ in 98.0:1:102.0]
	μs2  = [μ for μ in 98.0:0.1:102.0]
	ls1 = [logLikelihoodOfGaussianSumVersion(ys::Vector, μ=μ, σ=15.0) for μ in μs1]
	ls2 = [logLikelihoodOfGaussianSumVersion(ys::Vector, μ=μ, σ=15.0) for μ in μs2]
	plot(μs2, ls2, label=L"(μ, logL(\mu\;y,\sigma))", xlabel=L"\mu", ylabel=L"logL(\mu;y,\sigma)", title="log-Likelihoods (cf. Brown, 2014, Fig. 3.2, p.74)", titlefontsize=10)
	scatter!(μs1, ls1, label=L"(μ, logL(\mu\;y,\sigma))")
end # let

# ╔═╡ da81d6b1-ff8a-4928-923a-5041334389b9
md"
The following values of Gaussian log-likelihood are identical to those in Brown's Tab. 3.3 (Brown, 2014, Tab.3.3,  p.74)
"

# ╔═╡ 18dff2c8-0b08-46ec-9db4-cd684f19203a
logLikelihoodOfGaussianSumVersion(ys::Vector, μ=98.0, σ=15.0) 

# ╔═╡ 2c921e6e-8a79-4fd8-8ac4-1da4163eccbc
logLikelihoodOfGaussianSumVersion(ys::Vector, μ=99.0, σ=15.0) 

# ╔═╡ 5cb139c4-8fa4-4304-a7d0-7f3ea9f47811
logLikelihoodOfGaussianSumVersion(ys::Vector, μ=100.0, σ=15.0) 

# ╔═╡ 78adacdb-279f-4221-a1dc-dc3152e4fd2c
logLikelihoodOfGaussianSumVersion(ys::Vector, μ=101.0, σ=15.0) 

# ╔═╡ 95cc2784-7230-4d13-9ebc-6642b3609ff8
logLikelihoodOfGaussianSumVersion(ys::Vector, μ=102.0, σ=15.0) 

# ╔═╡ e45e1658-2ec0-48f8-9d4a-64a4806eafc4
md"
---
##### 3. ML (= Maximum Likelihood) Estimator $\hat{\mathbb \phi}_{MLE}$
Very plausible parameter values should have a high likelihood and less plausible a low likelihood. The most plausible value is the *maximum likelihood* value. In Bayesian statistics this is the *Maximum Aposterior (MAP)* value when the prior is uniform. 

###### 3.1 Definition

The MLE is invariant against logarithmic transformations which have algebraic and numeric advantages:

$\hat{\mathbb \phi}_{MLE} = \underset{\mathbb \phi}{\operatorname{argmax}}\mathcal\ L(\mathbb \phi|h, x_{1:n}, y_{1:n}) = \underset{\mathbb \phi}{\operatorname{argmax}}\mathcal\ l(\mathbb \phi|h, x_{1:n}, y_{1:n})$

$\;$
$\;$

$\hat{\mathbb \phi}_{MLE} = \underset{\mathbb \phi}{\operatorname{argmax}} \mathcal\  L(\mathbb \phi|h, x_{1:n}, y_{1:n}) = \underset{\mathbb \phi}{\operatorname{argmax}}\left(\prod_{i=1}^n f(y_i|h(x_i,\mathbb \phi), x_i, \phi)\right)$

$\;$
$\;$
$\;$

$\hat{\mathbb \phi}_{MLE} = \underset{\mathbb \phi}{\operatorname{argmax}}\ \mathcal l(\mathbb \phi|h, x_{1:n}, y_{1:n}) = \underset{\mathbb \phi}{\operatorname{argmax}}\left(log \prod_{i=1}^n f(y_i|h(x_i,\mathbb \phi), x_i, \phi)\right)$

$\;$
$\;$
$\;$

$\hat{\mathbb \phi}_{MLE} =  \underset{\mathbb \phi}{\operatorname{argmax}}\ \mathcal l(\mathbb \phi|h, x_{1:n}, y_{1:n}) = \underset{\mathbb \phi}{\operatorname{argmax}}\left(\sum_{i=1}^n log \left(f(y_i|h(x_i,\mathbb \phi), x_i, \phi\right)\right)$

$\;$
$\;$
$\;$

$\hat{\mathbb \phi}_{MLE} = \underset{\mathbb \phi}{\operatorname{argmin}}\left(-\sum_{i=1}^n log \left(f(y_i|h(x_i,\mathbb \phi), x_i, \phi\right)\right)$

$\;$
$\;$
$\;$

where:

$\hat{\phi}_{MLE} = \text{ ML estimator of parameter vector }\mathbb \phi$

$\;$
$\;$

$\mathcal l = \text{ log-likelihood function }$.

$\;$

"

# ╔═╡ fdf4c2ab-ba85-43c9-a33d-402979add2ab
md"
---
###### 3.2 Example: Grid Search of $\mu$ by Finding Extrema of Gaussian *log-likelihood*

###### 3.2.1 Example: Search for Extrema in Negative Gaussian *log-likelihood*
Either we look for the maximum or the minimum of relevant expressions.

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmax}}\mathcal\ l(\mathbb \mu; f_{\mathcal N}, \sigma, y_{1:n}) = \underset{\mathbb \mu}{\operatorname{argmax}}\left(- n \cdot log(\sqrt{2\pi\sigma^2}) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)=$

$\;$
$\;$
$\;$
$\;$

$= \underset{\mathbb \mu}{\operatorname{argmin}}\left(n \cdot log(\sqrt{2\pi\sigma^2}) + \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)=$

$\;$
$\;$
$\;$
$\;$

$= \underset{\mathbb \mu}{\operatorname{argmin}}\left(\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
$\;$

$= \underset{\mathbb \mu}{\operatorname{argmin}}\left(\sum_{i=1}^n (y_i-\mu)^2\right).$

$\;$
$\;$
$\;$
$\;$

So we see that under the assumption of some assumptions the MLE and the Least Squares Estimator (LSE) are identical.

"

# ╔═╡ 7b4a274e-8ade-415b-9da8-d1f3949622e7
md"
###### 3.2.2 Equality of Estimates by Maximzing and Minimizing Likelihood Kernels

The *log-likelhood kernel* is:

$l(\mu;f_{\mathcal N}, \sigma, y_{1:n})=- \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-\mu)^2$

$\;$
$\;$
$\;$
$\;$

and the *MLE*-estimator can be obtained by either maximizing the *log-likelihood kernel* or minimizing the *sum-of-squares*:

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmax}}\left(- \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-\mu)^2\right) = \underset{\mathbb \mu}{\operatorname{argmin}}\left(\frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
$\;$

"

# ╔═╡ 6e33b662-98a7-463f-9720-92bc571f026c
# hypthesized (guess) values for fixed (unknown) parameter μ
μs = [μ for μ in 98.0:0.1:102.0]  

# ╔═╡ 812cc0ff-765d-4fc7-9668-ecf2fcc50321
function findMinMax(foo::Function, ϕs::Array{Float64})
	#--------------------------------------------------------------------------------
	values = map(foo, ϕs)
	minima = minimum(values, init = +Inf)
	indicesMin = findall(logL -> logL == minima, values)
	maxima = maximum(values, init  =-Inf)
	indicesMax = findall(logL -> logL == maxima, values)
	argMins  = map(index -> ϕs[index], indicesMin)
	argMaxs  = map(index -> ϕs[index], indicesMax)
	#--------------------------------------------------------------------------------
	minLikelihoods = 
		(minimum=minima, indicesMin=indicesMin, argMins=argMins)
	maxLikelihoods = 
		(maximum=maxima, indicesMax=indicesMax, argMaxs=argMaxs)
	minLikelihoods, maxLikelihoods
end # function findMinMax

# ╔═╡ c14d7419-6c3f-4b0f-8f01-bc94d69feece
md"
###### 3.2.3 Example: Minima and Maximum of Gaussian Log-Likelihood Kernel

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmax}}\mathcal\ l(\mathbb \mu; f_{\mathcal N}, \sigma, y_{1:n}) = \underset{\mathbb \mu}{\operatorname{argmax}}\left(- \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
$\;$

"

# ╔═╡ cb602acc-7e5d-4cbc-9e75-bddce5e60258
function gaussianLogLikelihoodKernel(xs, μ; σ=15.0)
	- 1/(2.0*σ^2) * sum((xs .- μ).^2)
end # function gaussianLogLikelihoodKernel

# ╔═╡ 159a50b6-f89f-4ef4-8986-1906cb6ba1b0
findMinMax(μ -> gaussianLogLikelihoodKernel(ys::Vector, μ), μs)

# ╔═╡ 9c739f70-c972-4f5a-b587-ca40328b2549
md"
###### 3.2.4 Example: Minima and Maximum of Negative Gaussian Log-Likelihood Kernels

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmax}}\mathcal\ l(\mathbb \mu; f_{\mathcal N}, \sigma, y_{1:n}) = \underset{\mathbb \mu}{\operatorname{argmin}}\left(\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
$\;$

"

# ╔═╡ 5f579379-7d13-4db9-a9e0-6fd2093fd8f9
function negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	1/(2.0*σ^2) * sum((ys .- μ).^2)
end # function negativeGaussianLogLikelihoodKernel

# ╔═╡ 0bed8c9b-80fc-4d34-b221-fc5c0096a084
findMinMax(μ -> negativeGaussianLogLikelihoodKernel(ys::Vector, μ), μs)

# ╔═╡ 3b0e43e8-f851-4fe7-a854-b05d0b252e26
md"
---
###### 3.3 Example: Calculus Guided Search of $\mu_{MLE}$

###### 3.3.1 Example: Calculus Guided Solution: Fisher's *Score* Function $S(\phi)$
The first derivative of the log-likelihood function is called Fisher's *score* $S(\phi)$ function (Held & Bové, 2020, p.27).

Now, we want to use the score function to obtain the $\mu_{MLE}$. Under the assumption of known $\sigma$ it is sufficient to *maximize* the *log-likelhood kernel* or *minimize* the *sum-of-squares*:  

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmax}}\left(- \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right) = \underset{\mathbb \mu}{\operatorname{argmin}}\left(\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
$\;$

We determine the derivate of $S(\phi) = \frac{d \mathcal l(\phi)}{d\mu}$, set $S(\phi)=0$, solve the resulting equation to get $\mu_{MLE}$ which is identical to the LSE or OLS estimator:

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} = \frac{d \left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right)}{d \mu} = \frac{d \left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i^2-2y_i\mu+\mu^2)\right)}{d \mu}$

$\;$
$\;$
$\;$
$\;$

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} = \frac{d \left(-\frac{1}{2\sigma^2}\sum_{i=1}^n y_i^2\right)}{d \mu} + \frac{d \left(\frac{1}{2\sigma^2}\sum_{i=1}^n 2y_i\mu\right)}{d \mu} + \frac{d \left(-\frac{1}{2\sigma^2}\sum_{i=1}^n \mu^2\right)}{d \mu}$ 

$\;$
$\;$
$\;$
$\;$

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} = - 0 + 2\frac{1}{2\sigma^2}\left(\sum_{i=1}^n y_i\right) - 2\frac{1}{2\sigma^2}\sum_{i=1}^n\mu = \frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - μ\right)$

$\;$
$\;$
$\;$
$\;$

So the *score* is (Held & Bove, 2020, p.28):

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} = \frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - μ\right).$

$\;$
$\;$
$\;$
$\;$

"

# ╔═╡ 010000b3-dbbd-4614-a7ad-242ac2697155
md"
---
###### 3.3.2 Example: Solving the Score Equation $S(ϕ) = 0$

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} =\frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - μ\right) = 0$ 

$\;$
$\;$
$\;$

$\sum_{i=1}^n y_i - nμ = 0$

$\;$
$\;$
$\;$

$\sum_{i=1}^n y_i = nμ$

$\;$
$\;$
$\;$

$\hat \mu_{MLE} = \frac{1}{n}\sum_{i=1}^n y_i = \bar y$

$\;$
$\;$

where: 

$\;$

$\bar y = \text{ the sample mean }$

$\;$
$\;$
$\;$
"

# ╔═╡ 3b0f3cbd-0604-44bf-a6d4-ea4d9db43107
md"
---
###### 3.3.3 Score Function, Hessian, and Fisher Information

The *log-likelhood kernel* is:

$l(\mu;f_{\mathcal N}, \sigma, y_{1:n})=- \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-\mu)^2$

$\;$
$\;$
$\;$

and the *MLE*-estimator $\mu_{MLE}$ is:

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmax}}\left(- \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-\mu)^2\right) = \underset{\mathbb \mu}{\operatorname{argmin}}\left(\frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-\mu)^2\right)$

$\;$
$\;$
$\;$
$\;$

Fisher's *score* function is the first derivate of the log-likelihood function  (Held & Bove, 2020, p.28):

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} = \frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - μ\right).$

$\;$
$\;$
$\;$

The negative second derivative of the *log-likelihood* and the negative first derivative of the *score* function are called the *Fisher information*. The value of the Fisher information at the $\phi_{MLE}$ is called the *observed Fisher information* (Helde & Bové, 2020, p.27).

$I(\mu) = - \frac{d S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n})}{d \mu} = -\frac{d^2 \mathcal l}{d^2 \mu} = \frac{d\left(- \frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - \mu\right)\right)}{d \mu} =  - 0 + \frac{1}{\sigma^2}n = \frac{n}{\sigma^2}$

$\;$
$\;$
$\;$

$I(\mu) = \frac{n}{\sigma^2}\;;\;\text{ (Held \& Bovè, 2020, p.28)}$ 

$\;$
$\;$
$\;$

and here:

$I(\mu) = I(\hat \mu) = \text{ is the }\textit{observed}\text{ Information }$ 

$\;$
$\;$

The information $I(\mu)$ is *independent* of $\mu$ so it is independent of any statistic $\hat \mu$. In this case both *information* and *observed information* are identical.

The *Hessian* is the matrix with second derivatives (Brown, 2014, p.89). By taking expectations you can derive the *Fisher information*: 

$I = -E(H)$

$\;$

In this case the *Hessian* consists only of one element:

$H = \left[\frac{d^2 \mathcal l}{d^2 \mu}\right] = \left[-\frac{n}{\sigma^2}\right]$

$\;$
$\;$
$\;$

and

$I = -E(H) = -E\left(\left[-\frac{n}{\sigma^2}\right]\right)=\frac{n}{\sigma^2}$

$\;$
$\;$
$\;$

Here the Hessian is for all values $n,\sigma^2$ *negative definite* so the extremum is a *maximum*.

"

# ╔═╡ bc2f43b5-edc8-471a-aac8-3f8c0bbf9f0e
md"
---
###### 3.3.5 Fisher Information and Standard Error of $\mu$
(Brown, 2014, p.90)

$std_{error}(\hat \mu) = \sqrt{I^{-1}(\mu)} = \sqrt{\frac{1}{I(\mu)}} = \sqrt{\frac{1}{I(\hat \mu)}} = \sqrt{\frac{\sigma^2}{n}}= \frac{\sigma}{\sqrt n}$

$\;$
$\;$
$\;$
$\;$
"

# ╔═╡ 5558ef1e-2d5b-4397-bdaf-c593b0f421d4
begin 
	sqrtn = sqrt(n)
	ystderror_σ = 15.0/sqrtn
	ystderror_s = ystddev/sqrtn
	ystderror_σ, ystderror_s
end # begin

# ╔═╡ cbfcf521-1a2d-41c9-a796-39c5ffc13fc4
md"
---
###### 3.3.6 Standard Error and Confidence Limits $CL$ for $\mu$

Definition: *For repeated random samples from a distribution with unknown parameter $\phi$, a $\gamma \cdot 100 \%$ confidence interval will cover parameter $\phi$ in $\gamma \cdot 100 \%$ of all cases* (Held & Bové, 2020, p.57) $(\gamma \in (0,1))$:

$P(CL_{low} \le \phi \le CL_{high}) = \gamma$

$\;$
$\;$

$(CL_{low}, CL_{high}) = \hat \mu \pm t_{two-tailed, 0.05}*std_{error}(\hat \mu)$

$\;$
$\;$

The *confidence level* $\gamma$ is called the *coverage probability* for *fixed* (!) parameter $\phi$ and *random* $CL_{low}, \phi \le CL_{high}$:

$P(CL_{low} \le \mu \le CL_{high}) = \gamma = 0.95$

$\;$
$\;$

"

# ╔═╡ 41a626a6-1616-40ee-859f-ba323ae621f1
cdf(TDist(12), -2.1788), cdf(TDist(12), 2.1788)

# ╔═╡ 94391f3b-22f8-40c0-b0dd-66dbcc9d0d34
cl_σ = (ybar - 2.1788 * ystderror_σ, ybar + 2.1788 * ystderror_σ)

# ╔═╡ b3574040-fd96-43cd-b3c9-05d1d36956c8
cl_s = (ybar - 2.1788 * ystderror_s, ybar + 2.1788 * ystderror_s)

# ╔═╡ d79d1e1a-4865-456d-9da5-ab2d11283b58
md"
---
###### Example: 3.3.7 Numerical Minimization of Negative Log-likelihood Kernel
###### Example: 3.3.7.1 Derivate-free Minimization with Nelder-Mead
Some algorithms (e.g. Nelder-Mead) only *minimize* functions. So we provide the *negative log-likelihood kernel* here: 

$\hat \mu_{MLE} = \underset{\mathbb \mu}{\operatorname{argmin}}\left(\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-\mu)^2\right) = 100.0 = \bar x$

$\;$
$\;$
$\;$

"

# ╔═╡ 34ff0aa9-2fae-4afb-bf85-acac6d700ef4
ys

# ╔═╡ c6ac6881-64ce-4414-af59-2f0fcb23577d
let nGLLK(μ) = negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	μ0   = [ 75.0]
	results = optimize(nGLLK, μ0, NelderMead())
	Optim.minimizer(results), Optim.minimum(results), results
end # let

# ╔═╡ 91550cbb-5cb6-491e-9459-aeee6120b8a7
md"
---
###### Example: 3.3.7.2 Minimization with [L-BFGS](https://julianlsolvers.github.io/Optim.jl/stable/user/minimization/) Using Gradients

"

# ╔═╡ c14e5edf-6277-4d8b-b92b-a2e79dd6c0f9
md"
###### *L-BFGS* with [Approximate Gradient Using Central Finite Differencing](https://julianlsolvers.github.io/Optim.jl/stable/user/minimization/)
"

# ╔═╡ 4bac2323-7a2f-4797-9542-6bb5d45c0dd3
let nGLLK(μ) = negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	μ0   = [ 75.0]
	results = optimize(nGLLK, μ0, LBFGS())
	Optim.minimizer(results), Optim.minimum(results), results
end # let

# ╔═╡ 6727ce62-a589-4351-862f-adc01ce30307
md"
###### *L-BFGS* with [Automatic Differentiation triggered by Keyword](https://julianlsolvers.github.io/Optim.jl/stable/user/minimization/) $autodiff$
"

# ╔═╡ 62567f28-3100-4762-9f08-a9a596062bf8
let nGLLK(μ) = negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	μ0   = [ 75.0]
	results = optimize(nGLLK, μ0, LBFGS(); autodiff = :forward)
	Optim.minimizer(results), Optim.minimum(results), results
end # let

# ╔═╡ 1f7d29f1-f951-4d82-9bf4-7901d4eeb51f
md"
###### *L-BFGS* with Explicit Information about Gradients (= *Score* Function)
The *score* is (Held & Bove, 2020, p.28):

$S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = \frac{d \mathcal l}{d \mu} = \frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - μ\right)$

$\;$
$\;$
$\;$

For algorithm $LBFGS$ we had to switch from *positive* to *negative* score:

$-S(\mathbb \mu; \mathcal l_{\mathcal N}, \sigma, y_{1:n}) = -\frac{d \mathcal l}{d \mu} = -\frac{1}{\sigma^2}\left(\sum_{i=1}^n y_i - μ\right).$

$\;$
$\;$
$\;$
"

# ╔═╡ 63282a7b-554c-46d4-9762-375dfeba6b9f
# no inplace mutation of gradient
#----------------------------------------------------------------
let nGLLK(μ) = negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	#------------------------------------------------------------
	negativeScore(μ; σ=15.0) = -(1.0/σ^2)*sum(ys .- μ)
	#------------------------------------------------------------
	function g(μ) 
		negativeScore(μ; σ=15.0)
	end # function g
	#------------------------------------------------------------
	μ0   = [ 75.0]
	results = optimize(nGLLK, g, μ0, LBFGS(); inplace=false)
	Optim.minimizer(results), Optim.minimum(results), results
	#------------------------------------------------------------
end # let

# ╔═╡ a57321e6-faef-4786-b03a-8f0df6a655d9
# inplace mutation of gradient G
#----------------------------------------------------------------
let nGLLK(μ) = negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	#------------------------------------------------------------
	negativeScore(μ; σ=15.0) = -(1.0/σ^2)*sum(ys .- μ)
	#------------------------------------------------------------
	function g!(G, μ) 
		G[1] = negativeScore(μ; σ=15.0)
	end # function g!
	#------------------------------------------------------------
	μ0   = [ 75.0]
	results = optimize(nGLLK, g!, μ0, LBFGS())
	Optim.minimizer(results), Optim.minimum(results), results
	#------------------------------------------------------------
end # let

# ╔═╡ fee13909-fea3-4b8e-971e-7632766b3c5c
md"
###### *Newton* Algorithm with Explicit Gradients (= *Score* Function) and *Hessians* 
"

# ╔═╡ 7d651c60-8f75-4428-ae91-2b933e23b7cf
# inplace mutation of Hessian H
#----------------------------------------------------------------
let nGLLK(μ) = negativeGaussianLogLikelihoodKernel(ys, μ; σ=15.0)
	#------------------------------------------------------------
	negativeScore(μ; σ=15.0) = -(1.0/σ^2)*sum(ys .- μ)
	#------------------------------------------------------------
	function g!(G, μ; σ=15.0) 
		G[1] = negativeScore(μ; σ=15.0)
	end # function g!
	#------------------------------------------------------------
	function h!(H, μ; σ=15.0)
		H[1, 1] = - n/σ^2
	end # function h!
	#------------------------------------------------------------
	μ0   = [ 75.0]
	results = optimize(nGLLK, g!, h!, μ0, Newton())
	Optim.minimizer(results), Optim.minimum(results), results
	#------------------------------------------------------------
end # let

# ╔═╡ 4943eec5-b299-4e70-a80a-bcf407c35074
md"
---
##### 4. Summary
We presented and used various *MLE*-related concepts in the context of the Gaussian distributed random samples: density, likelihood, density and likelhood function, log-likelihood, likelihood kernel, minimization and maximization, first and second derivatives of log-likelihood function, Fisher's score function, Hessian, Fisher information, Nelder-Mead-, L-BFGS-, and Newton-algorithm.

Besides several others we used function of the Optim.jl package.
"

# ╔═╡ d6051c16-4b38-4784-a7b8-0150027294cb
md"
---
##### References

- **Bishop, C.M. & Bishop, H.**; *Deep Learning: Foundations and Concepts*, Cham, Swiss: Springer, 2024

- **Brown, J.D.**; *Linear Models in Matrix Form*, Cham: Springer, 2014

- **Held, L. & Bové, D.S.**; *Likelihood and Bayesian Inference*, Berlin: Springer, 2nd/e, 2020

- **Prince, S.J.D.**; [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/); MIT Press, 2024, last visit 2024/07/05

- **Prince, S.J.D.**; [*Notebook 5.1 – Least Squares Loss*](https://udlbook.github.io/udlbook/); last visit 2024/07/05

- **Wikipedia**; [*Probability density function*](https://en.wikipedia.org/wiki/Probability_density_function); last visit 2024/07/08

- **Wikipedia**; [*Probability mass function*](https://en.wikipedia.org/wiki/Probability_mass_function); last visit 2024/07/08

"


# ╔═╡ afb4fc4d-8d81-40c7-bd3d-40575f881a34
md"
====================================================================================

This is a **draft** under the Attribution-NonCommercial-ShareAlike 4.0 International **(CC BY-NC-SA 4.0)** license. Comments, improvement and issue reports are welcome: **claus.moebus(@)uol.de**

===================================================================================
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
Distributions = "~0.25.107"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
Optim = "~1.9.2"
Plots = "~1.40.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "6a808c0671c545db1c4dcdaa38acc33b6242ae5a"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

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

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

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
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

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

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

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

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "569a003f93d7c64068d3afaab908d21f67a22cd5"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.25.3"

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

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "302fd161eb1c439e4115b51ae456da4e9984f130"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "MathOptInterface", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d024bfb56144d947d4fafcd9cb5cafbe3410b133"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.2"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

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

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

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

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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
# ╟─400e7456-8a3b-44cc-ac77-5b28087b81fb
# ╟─e2666b94-6c11-4de3-9b68-ed61a01aa334
# ╟─19818803-5892-46a4-b0a7-2025cd0555ba
# ╠═007be6ce-ca18-4877-82e4-f49f85fd993b
# ╟─3a2a4fca-9e51-4882-967a-4d62718dae94
# ╟─2d21c9e7-afb3-4354-8bb0-36dd672b3dcb
# ╟─85caf5f0-d836-4cb8-a078-0894ae07016d
# ╠═f56ef177-68fc-4cb0-94aa-4b8254b62406
# ╠═e116bbe9-bca6-4f7b-94cb-fdafa52f6ddd
# ╠═7c4bd2ef-b479-4708-ac63-90ee118bea47
# ╠═f94fc7fb-9f21-4c99-a0f4-ddf4a4df37c9
# ╠═95af722f-9675-4562-a115-166bb5e27679
# ╠═25584038-b162-438c-acc7-597fb5b65b0d
# ╟─479adfc6-00da-4279-b64d-9f8ef68d7619
# ╠═303aaffa-09aa-4c80-a92d-90045a7b688f
# ╠═127347fa-1ac3-4306-acc9-61091538ad71
# ╟─23e54ade-e9b0-47ae-91d4-fde9b5e3ec67
# ╟─54e75a6a-8ea9-41dc-ae12-e3691e0f893e
# ╟─38ac97c9-4a83-4740-8444-2fac23fe31ce
# ╠═2ea367b2-cedd-4720-a487-1dfacd1ed91a
# ╠═bbb2c7b7-1f2a-4daf-9bc3-dff8e7eb5e22
# ╟─50c9f7e8-1e09-4718-b012-aad4031bbfa0
# ╠═3b93960e-1f23-4be1-a1e9-05761d6bd263
# ╠═1e0f445a-4bf1-41db-b0c8-9f31b734cf8e
# ╠═4957f59d-eaff-4719-b3a2-b4b4b785411c
# ╠═c747204a-9ffe-4344-a5b6-214a30e95bb0
# ╠═00467aab-cfbc-4ce3-a02e-620eb32251cc
# ╟─6b5c88e9-4869-4820-972f-71f0988bed35
# ╠═546d9ae5-3104-4124-862c-f09f16d61806
# ╠═77f301fc-1e9e-49a4-8549-9d0918a2bd6e
# ╟─a94453a2-7724-4a47-a952-9efed4bdd197
# ╠═6d01b17d-7a2a-4689-91aa-3a941c22fdfa
# ╠═79527167-096e-47f3-b67f-38724ab7bab3
# ╠═67daba69-328e-4792-b6cc-65ab9de4096a
# ╠═213fc124-d74e-44aa-9c6e-e5b314908c2f
# ╠═97b79e30-1531-4cc9-887b-5b8549387b0f
# ╟─856646cb-6fdc-4acf-bfe1-ba4c110b7710
# ╠═84bc6149-3c9f-467e-9383-219b6b8a9ac6
# ╟─dee1689b-191b-44a2-8fd4-767e2a3d4f63
# ╠═229dba96-9b89-4f98-970a-62b21823d912
# ╠═98cb5124-dfd6-456d-b72d-24fd8732cebb
# ╠═aff7988f-b8e5-4207-905b-a29bc4ddae9a
# ╟─da81d6b1-ff8a-4928-923a-5041334389b9
# ╠═18dff2c8-0b08-46ec-9db4-cd684f19203a
# ╠═2c921e6e-8a79-4fd8-8ac4-1da4163eccbc
# ╠═5cb139c4-8fa4-4304-a7d0-7f3ea9f47811
# ╠═78adacdb-279f-4221-a1dc-dc3152e4fd2c
# ╠═95cc2784-7230-4d13-9ebc-6642b3609ff8
# ╟─e45e1658-2ec0-48f8-9d4a-64a4806eafc4
# ╟─fdf4c2ab-ba85-43c9-a33d-402979add2ab
# ╟─7b4a274e-8ade-415b-9da8-d1f3949622e7
# ╠═6e33b662-98a7-463f-9720-92bc571f026c
# ╠═812cc0ff-765d-4fc7-9668-ecf2fcc50321
# ╟─c14d7419-6c3f-4b0f-8f01-bc94d69feece
# ╠═cb602acc-7e5d-4cbc-9e75-bddce5e60258
# ╠═159a50b6-f89f-4ef4-8986-1906cb6ba1b0
# ╟─9c739f70-c972-4f5a-b587-ca40328b2549
# ╠═5f579379-7d13-4db9-a9e0-6fd2093fd8f9
# ╠═0bed8c9b-80fc-4d34-b221-fc5c0096a084
# ╟─3b0e43e8-f851-4fe7-a854-b05d0b252e26
# ╟─010000b3-dbbd-4614-a7ad-242ac2697155
# ╟─3b0f3cbd-0604-44bf-a6d4-ea4d9db43107
# ╟─bc2f43b5-edc8-471a-aac8-3f8c0bbf9f0e
# ╠═5558ef1e-2d5b-4397-bdaf-c593b0f421d4
# ╟─cbfcf521-1a2d-41c9-a796-39c5ffc13fc4
# ╠═41a626a6-1616-40ee-859f-ba323ae621f1
# ╠═94391f3b-22f8-40c0-b0dd-66dbcc9d0d34
# ╠═b3574040-fd96-43cd-b3c9-05d1d36956c8
# ╟─d79d1e1a-4865-456d-9da5-ab2d11283b58
# ╠═34ff0aa9-2fae-4afb-bf85-acac6d700ef4
# ╠═c6ac6881-64ce-4414-af59-2f0fcb23577d
# ╟─91550cbb-5cb6-491e-9459-aeee6120b8a7
# ╟─c14e5edf-6277-4d8b-b92b-a2e79dd6c0f9
# ╠═4bac2323-7a2f-4797-9542-6bb5d45c0dd3
# ╟─6727ce62-a589-4351-862f-adc01ce30307
# ╠═62567f28-3100-4762-9f08-a9a596062bf8
# ╟─1f7d29f1-f951-4d82-9bf4-7901d4eeb51f
# ╠═63282a7b-554c-46d4-9762-375dfeba6b9f
# ╠═a57321e6-faef-4786-b03a-8f0df6a655d9
# ╟─fee13909-fea3-4b8e-971e-7632766b3c5c
# ╠═7d651c60-8f75-4428-ae91-2b933e23b7cf
# ╟─4943eec5-b299-4e70-a80a-bcf407c35074
# ╟─d6051c16-4b38-4784-a7b8-0150027294cb
# ╟─afb4fc4d-8d81-40c7-bd3d-40575f881a34
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
