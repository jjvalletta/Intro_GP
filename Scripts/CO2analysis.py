#----------------------------------------------------------------------------------------------------------------------#
# Title:    Model CO2 data
# Author: 	John Joseph Valletta
# Date:     12/09/2015	 
# Data:     http://co2now.org/images/stories/data/co2-mlo-monthly-noaa-esrl.xls
# From:     Mauna Loa Observatory, Hawaii
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Preamble
#----------------------------------------------------------------------------------------------------------------------#
# Libraries
import pandas # read .csv as a data frame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # to save as pdf
import GPy
# Constants
FILE_LOC = "/Users/jjv207/Machine_Learning/Intro_GP/Data/CO2Data.csv"
RESULTS_LOC = "/Users/jjv207/Machine_Learning/Intro_GP/Figures/"
XLIM = (1957, 2025)
YLIM = (310, 425)

#----------------------------------------------------------------------------------------------------------------------#
# Load and plot data
# -99.99: missing data
# -1    : number of data days is not available (number of days to average CO2)
#----------------------------------------------------------------------------------------------------------------------#
data = pandas.read_csv(FILE_LOC, sep=",")
bWant = data.CO2 > 0
yTrain = data.CO2[bWant][:,None]
xTrain = data.date[bWant][:,None]
hFig = plt.figure("CO2 vs Year")
plt.plot(xTrain, yTrain, '.', ms=2) # ms = markersize
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.xlabel("Year")
plt.ylabel("CO$_2$ concentration (ppm)")
# Save figure
fileName = "CO2Raw.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure

#----------------------------------------------------------------------------------------------------------------------#
# Fit polynomials
#----------------------------------------------------------------------------------------------------------------------#
xTest = np.linspace(start=XLIM[0], stop=XLIM[1], num=1000)[:,None]
# First order
fit = np.poly1d(np.polyfit(x=xTrain, y=yTrain, deg=1)) # polyfit returns the polynomial coefficients
hFig = plt.figure("Poly fit deg=1")
plt.plot(xTrain, yTrain, '.', ms=2) # ms = markersize
plt.plot(xTest, fit(xTest), color="red")
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.xlabel("Year")
plt.ylabel("CO$_2$ concentration (ppm)")
# Save figure
fileName = "CO2PolyFit1.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure
# Second order
fit = np.poly1d(np.polyfit(x=xTrain, y=yTrain, deg=2)) # polyfit returns the polynomial coefficients
hFig = plt.figure("Poly fit deg=2")
plt.plot(xTrain, yTrain, '.', ms=2) # ms = markersize
plt.plot(xTest, fit(xTest), color="red")
plt.xlim(XLIM)
plt.ylim(YLIM)
plt.xlabel("Year")
plt.ylabel("CO$_2$ concentration (ppm)")
# Save figure
fileName = "CO2PolyFit2.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure

#----------------------------------------------------------------------------------------------------------------------#
# Gaussian Process Fit on CO2 data
#----------------------------------------------------------------------------------------------------------------------#
xTest = np.linspace(start=1950, stop=2035, num=5000)[:,None]
# Plot raw data again first
hFig = plt.figure("CO2 vs Year Take 2", figsize=(16,6))
plt.plot(xTrain, yTrain, '.', ms=2) # ms = markersize
plt.xlim(min(xTest), max(xTest))
plt.ylim(300, 440)
plt.xlabel("Year")
plt.ylabel("CO$_2$ concentration (ppm)")
# Save figure
fileName = "CO2RawTake2.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure
# Now do the GP fit
# Took parameters from Rasmussen book page 119 
# (it's only an approximation but good enough to convey the message)
cov1 = GPy.kern.RBF(input_dim=1, variance=70**2, lengthscale=67) # covFunction.plot() # plot of the covariance fn
cov2 = GPy.kern.PeriodicExponential(input_dim=1, variance=20, lengthscale=90, period=3)
cov3 = GPy.kern.RatQuad(input_dim=1, variance=0.66, lengthscale=1.2*0.78, power=0.78)
cov4 = GPy.kern.RBF(input_dim=1, variance=0.04, lengthscale=1.6)
covFunction = cov1 + cov2 + cov3 + cov4
fit = GPy.models.GPRegression(X=xTrain, Y=yTrain, kernel=covFunction)
fit.Gaussian_noise.variance = 0.5
#fit.optimize_restarts(num_restarts=10)
# Compute the mean function and plot
mu, var = fit.predict(Xnew=xTest)
# Plot the mean function and data points
hFig = plt.figure("GPFit", figsize=(16,6))
plt.plot(xTrain, yTrain, '.', ms=2)
plt.xlim(min(xTest), max(xTest))
plt.ylim(300, 440)
plt.xlabel("Year")
plt.ylabel("CO$_2$ concentration (ppm)")
# Plot bounds
hAx = plt.gca()
GPy.plotting.matplot_dep.base_plots.gpplot(x=xTest, mu=mu, lower=mu-2.*np.sqrt(var), upper=mu+2.*np.sqrt(var), 
                                               edgecol="black", fillcol="grey", ax=hAx, alpha=0.3)
# Save figure
fileName = "CO2GPFit.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure                                                                                   