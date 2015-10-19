#----------------------------------------------------------------------------------------------------------------------#
# Title:   GP toy examples
# Author: 	John Joseph Valletta
# Date:    08/10/2015	 
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Preamble
#----------------------------------------------------------------------------------------------------------------------#
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # to save as pdf
from scipy.stats import multivariate_normal
from matplotlib import cm # colourmaps
import GPy # to get Gaussian Prior
from brewer2mpl import qualitative # colour brewer colours
# Constants
RESULTS_LOC = "/Users/jjv207/MachineLearning/Intro_GP/Figures/"
COLOUR = qualitative.Set1[4].hex_colors
YLIM = (-2.5, 2.5)
SHADE_COLOUR = "black" # "darkgrey"
LINE_WIDTH_MU = 6 # line width for mean function
LINE_WIDTH_SAMPLES = 4 # line width for posterior samples
ALPHA = 0.3
    
#----------------------------------------------------------------------------------------------------------------------#
# Define Functions
#----------------------------------------------------------------------------------------------------------------------#
def setlimitsandsave(fileName="Untitled.pdf", hFig=plt.gcf(), ylim=(-100, 100), xlim=(-100, 100)):
    """
    A bit of an ugly function but avoids me having to write the same bit of code every time I want to save a figure

    Arguments
    =========
    organ       - "blood" or "spleen" only
    strain      - "AS" or "CB" only
    topRanked   - how many top ranked differentially expressed genes are considered for clustering

    """
    # Set limits and save
    plt.ylim(ylim)                                
    plt.xlim(xlim)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("$f(x)$", fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    # Save figure
    with PdfPages(RESULTS_LOC + fileName) as pdf:     
            pdf.savefig(hFig) # save figure
            plt.close(hFig) # close figure
            
#----------------------------------------------------------------------------------------------------------------------#
# Generate some fictitious data
#----------------------------------------------------------------------------------------------------------------------#
np.random.seed(143) # to replicate result
N = 40 # no. of samples
xTest = np.linspace(start=0, stop=4*np.pi, num=200)[:, None] # test points for predictions
xTrain = np.random.uniform(low=0, high=4*np.pi, size=N)[:, None]
y = np.sin(xTrain)
yTrain = y + np.random.normal(loc=0, scale=0.2, size=y.shape) # loc=mean, scale=stdev
plt.plot(xTrain, yTrain, 'ko')

#----------------------------------------------------------------------------------------------------------------------#
# Gaussian Prior
#----------------------------------------------------------------------------------------------------------------------#
# Specify mean function and covariance function
np.random.seed(120) # to replicate result
mean = np.zeros((len(xTest)))
covFunction = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=np.pi/2) # covFunction.plot() # plot of the covariance fn
covMatrix = covFunction.K(xTest, xTest)
yTest = np.random.multivariate_normal(mean, covMatrix, len(COLOUR)) # just take 1 sample, but can change that to more
# Set-up plot area
fileName = "GPPrior"
hFig = plt.figure(fileName)
hAx = plt.gca()
for iPrior in xrange(len(COLOUR)):
    mu = yTest[iPrior,:] # a sample from the GP prior
    plt.plot(xTest, mu, COLOUR[iPrior], lw=LINE_WIDTH_SAMPLES)
# Use the brilliant plotting function from GPy to plot bounds (no need to re-invent the wheel)      
var = covFunction.variance # variance in this case is just a single number
GPy.plotting.matplot_dep.base_plots.gpplot(x=xTest, mu=mu, lower=-2*var*np.ones(mu.shape), upper=+2*var*np.ones(mu.shape), 
                                               edgecol=COLOUR[iPrior], fillcol=SHADE_COLOUR, ax=hAx, alpha=ALPHA)
# Plot the mean function
plt.plot(xTest, mean, color="black", lw=LINE_WIDTH_MU)
# Save plot
setlimitsandsave(fileName=fileName + ".pdf", hFig=hFig, ylim=YLIM, xlim=(min(xTest), max(xTest)))

#----------------------------------------------------------------------------------------------------------------------#
# Start adding points and plot
#----------------------------------------------------------------------------------------------------------------------#
np.random.seed(267) # to replicate result
# Initialise iiSelect i.e the data I've observed so far
# choiceVector the indices that I can pick from i.e those which I didn't observe yet
iiSelect = np.array([], dtype=int)
choiceVector = np.arange(start=0, stop=len(xTrain))
for iPoint in xrange(len(xTrain)):
    temp = np.random.choice(a=choiceVector, size=1, replace=False)
    iiSelect = np.append(iiSelect, temp)
    choiceVector = np.delete(choiceVector, temp)
    # Fit model
    fit = GPy.models.GPRegression(X=xTrain[iiSelect, :], Y=yTrain[iiSelect, :], kernel=covFunction)
    fit.Gaussian_noise.variance = 0.2**2 # set this else default is too high
    # Set-up plot area
    fileName = "GP" + str(iPoint) + "Point"
    hFig = plt.figure(fileName)
    hAx = plt.gca()
    # Compute samples from the posterior
    yPosterior = fit.posterior_samples_f(xTest, size=len(COLOUR)) 
    for iPost in xrange(len(COLOUR)):
        plt.plot(xTest, yPosterior[:, iPost], color=COLOUR[iPost], lw=3)
    # Compute the mean function and plot
    mu, var = fit.predict(Xnew=xTest)
    # Plot the mean function and data points
    plt.plot(xTest, mu, color="black", lw=LINE_WIDTH_MU)
    plt.plot(xTrain[iiSelect, :], yTrain[iiSelect, :], 'kx', ms=20, mew=4) # markeredgewidth
    # Plot bounds
    GPy.plotting.matplot_dep.base_plots.gpplot(x=xTest, mu=mu, lower=mu-2.*np.sqrt(var), upper=mu+2.*np.sqrt(var), 
                                                   edgecol="black", fillcol=SHADE_COLOUR, ax=hAx, alpha=ALPHA)
    # Save plot                                
    setlimitsandsave(fileName=fileName + ".pdf", hFig=hFig, ylim=YLIM, xlim=(min(xTest), max(xTest)))
    
#----------------------------------------------------------------------------------------------------------------------#
# Misspecifying Covariance Function
#----------------------------------------------------------------------------------------------------------------------#
# Fit model
np.random.seed(267) # to replicate result
xTest = np.linspace(start=-np.pi/2, stop=4*np.pi+np.pi/2, num=200)[:, None] # test points for predictions
iiSelect = np.random.choice(a=len(xTrain), size=20, replace=False) # Pick 20 points to make plot more visually appealing
lambdas = np.array([0.0625*np.pi, 0.125*np.pi, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 5*np.pi])
titleString = ["$\pi/16$", "$\pi/8$", "$\pi/4$", "$\pi/2$", "$3\pi/4$", "$\pi$", "$1.5\pi$", "$2\pi$", "$5\pi$"]
iCnt = 0
for iLambda in lambdas:
    covFunction = GPy.kern.RBF(input_dim=1, variance=0.6, lengthscale=iLambda) # covFunction.plot() # plot of the covariance fn
    fit = GPy.models.GPRegression(X=xTrain[iiSelect, :], Y=yTrain[iiSelect, :], kernel=covFunction)
    fit.Gaussian_noise.variance = 0.2**2 # set this else default is too high
    # Set-up plot area
    fileName = "GP"  + "Lengthscale" + str(iCnt)
    hFig = plt.figure(fileName)
    hAx = plt.gca()
    # Compute the mean function and plot
    mu, var = fit.predict(Xnew=xTest)
    # Plot the mean function and data points
    plt.plot(xTest, mu, color="black", lw=LINE_WIDTH_MU)
    plt.plot(xTrain[iiSelect, :], yTrain[iiSelect, :], 'kx', ms=20, mew=4) # markeredgewidth
    plt.title("Lengthscale = %s" % titleString[iCnt])    
    # Plot bounds
    GPy.plotting.matplot_dep.base_plots.gpplot(x=xTest, mu=mu, lower=mu-2.*np.sqrt(var), upper=mu+2.*np.sqrt(var), 
                                                   edgecol="black", fillcol=SHADE_COLOUR, ax=hAx, alpha=ALPHA)
    # Save plot                                
    setlimitsandsave(fileName=fileName + ".pdf", hFig=hFig, ylim=YLIM, xlim=(min(xTest), max(xTest)))
    # Update counter
    iCnt += 1
    