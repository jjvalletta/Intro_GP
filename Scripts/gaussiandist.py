#----------------------------------------------------------------------------------------------------------------------#
# Title:    Generate some Gaussians to explain Gaussian Processes
# Author: 	John Joseph Valletta
# Date:     12/09/2015	 
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Preamble
#----------------------------------------------------------------------------------------------------------------------#
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # to save as pdf
from mpl_toolkits.mplot3d import Axes3D # 3D projection
from matplotlib import cm # colourmaps
from scipy.stats import multivariate_normal
import GPy # to get Gaussian Prior
# Constants
RESULTS_LOC = "/Users/jjv207/MachineLearning/Intro_GP/Figures/"

#----------------------------------------------------------------------------------------------------------------------#
# Generate 1D Gaussian
#----------------------------------------------------------------------------------------------------------------------#
# 1D Gaussian Distribution
mean = np.array([0])
covMatrix = np.array([1])
xTest = np.linspace(start=-4, stop=4, num=400)
gauss = multivariate_normal(mean, covMatrix)
# Plot Gaussian Distribution
hFig = plt.figure("1D Gaussian")
plt.plot(xTest, gauss.pdf(xTest), lw=6, color="black")
plt.xlabel("$x$", fontsize=24)
plt.ylabel("pdf($x$)", fontsize=24)
plt.tick_params(axis='both', labelsize=16)
# Save figure
fileName = "Gaussian1D.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure
        
        
#----------------------------------------------------------------------------------------------------------------------#
# Generate 2D Gaussian
#----------------------------------------------------------------------------------------------------------------------#
# 2D Gaussian Distribution
mean = np.array([0, 0])
covMatrix = np.array([[1, 0], [0, 1]])
x, y = np.mgrid[-2:2:.01, -2:2:.1]
xTest = np.dstack((x, y))
gauss = multivariate_normal(mean, covMatrix)
# Plot Gaussian Distribution
hFig = plt.figure("2D Gaussian")
hAx = hFig.gca(projection='3d')
surf = hAx.plot_surface(x, y, gauss.pdf(xTest), rstride=1, cstride=1, cmap=cm.Greys, linewidth=0, antialiased=False)
hAx.set_xlabel("$x_1$", fontsize=24)
hAx.set_ylabel("$x_2$", fontsize=24)
hAx.set_zlabel("pdf($x_1$, $x_2$)", fontsize=24)
hAx.tick_params(axis='both', labelsize=8)
#hFig.colorbar(surf, shrink=0.5, aspect=5)
# Save figure
fileName = "Gaussian2D.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure

#----------------------------------------------------------------------------------------------------------------------#
# Generate 2D Gaussian - Isotropic
#----------------------------------------------------------------------------------------------------------------------#
# 2D Gaussian Distribution
mean = np.array([0, 0])
covMatrix = np.array([[1, 0], [0, 1]])
x, y = np.mgrid[-2:2:.01, -2:2:.1]
xTest = np.dstack((x, y))
gauss = multivariate_normal(mean, covMatrix)
# Plot Gaussian Distribution
hFig = plt.figure("2D Gaussian Contour")
CS = plt.contourf(x, y, gauss.pdf(xTest), cmap=cm.Greys) # alpha=0.5
#CS2 = plt.contour(CS, levels=CS.levels[::2], colors = 'r', hold='on')
#plt.clabel(CS, inline=1, fontsize=18) # use this with contour but not contourf
plt.ylim(-2.1, 2.1)
plt.xlim(-2.1, 2.1)
plt.xlabel("$x_1$", fontsize=24)
plt.ylabel("$x_2$", fontsize=24)
plt.tick_params(axis='both', labelsize=16)
plt.axis('equal')
#hFig.colorbar(surf, shrink=0.5, aspect=5)
# Save figure
fileName = "Gaussian2DContour1.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure
        
#----------------------------------------------------------------------------------------------------------------------#
# Generate 2D Gaussian - Diagonal
#----------------------------------------------------------------------------------------------------------------------#
# 2D Gaussian Distribution
mean = np.array([0, 0])
covMatrix = np.array([[1, 0], [0, 0.5]])
x, y = np.mgrid[-2:2:.01, -2:2:.1]
xTest = np.dstack((x, y))
gauss = multivariate_normal(mean, covMatrix)
# Plot Gaussian Distribution
hFig = plt.figure("2D Gaussian Contour")
CS = plt.contourf(x, y, gauss.pdf(xTest), cmap=cm.Greys) # alpha=0.5
#CS2 = plt.contour(CS, levels=CS.levels[::2], colors = 'r', hold='on')
#plt.clabel(CS, inline=1, fontsize=18) # use this with contour but not contourf
plt.ylim(-2.1, 2.1)
plt.xlim(-2.1, 2.1)
plt.xlabel("$x_1$", fontsize=24)
plt.ylabel("$x_2$", fontsize=24)
plt.tick_params(axis='both', labelsize=16)
plt.axis('equal')
#hFig.colorbar(surf, shrink=0.5, aspect=5)
# Save figure
fileName = "Gaussian2DContour2.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure

#----------------------------------------------------------------------------------------------------------------------#
# Generate 2D Gaussian - Generic +ve relationship
#----------------------------------------------------------------------------------------------------------------------#
# 2D Gaussian Distribution
mean = np.array([0, 0])
covMatrix = np.array([[1, 0.5], [0.5, 1]])
x, y = np.mgrid[-2:2:.01, -2:2:.1]
xTest = np.dstack((x, y))
gauss = multivariate_normal(mean, covMatrix)
# Plot Gaussian Distribution
hFig = plt.figure("2D Gaussian Contour")
CS = plt.contourf(x, y, gauss.pdf(xTest), cmap=cm.Greys) # alpha=0.5
#CS2 = plt.contour(CS, levels=CS.levels[::2], colors = 'r', hold='on')
#plt.clabel(CS, inline=1, fontsize=18) # use this with contour but not contourf
plt.ylim(-2.1, 2.1)
plt.xlim(-2.1, 2.1)
plt.xlabel("$x_1$", fontsize=24)
plt.ylabel("$x_2$", fontsize=24)
plt.tick_params(axis='both', labelsize=16)
plt.axis('equal')
#hFig.colorbar(surf, shrink=0.5, aspect=5)
# Save figure
fileName = "Gaussian2DContour3.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure
        
#----------------------------------------------------------------------------------------------------------------------#
# Generate 2D Gaussian - Generic -ve relationship
#----------------------------------------------------------------------------------------------------------------------#
# 2D Gaussian Distribution
mean = np.array([0, 0])
covMatrix = np.array([[1, -0.5], [-0.5, 1]])
x, y = np.mgrid[-2:2:.01, -2:2:.1]
xTest = np.dstack((x, y))
gauss = multivariate_normal(mean, covMatrix)
# Plot Gaussian Distribution
hFig = plt.figure("2D Gaussian Contour")
CS = plt.contourf(x, y, gauss.pdf(xTest), cmap=cm.Greys) # alpha=0.5
#CS2 = plt.contour(CS, levels=CS.levels[::2], colors = 'r', hold='on')
#plt.clabel(CS, inline=1, fontsize=18) # use this with contour but not contourf
plt.ylim(-2.1, 2.1)
plt.xlim(-2.1, 2.1)
plt.xlabel("$x_1$", fontsize=24)
plt.ylabel("$x_2$", fontsize=24)
plt.tick_params(axis='both', labelsize=16)
plt.axis('equal')
#hFig.colorbar(surf, shrink=0.5, aspect=5)
# Save figure
fileName = "Gaussian2DContour4.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure

#----------------------------------------------------------------------------------------------------------------------#
# Sample from a 100-dimensional Gaussian
#----------------------------------------------------------------------------------------------------------------------#
np.random.seed(123) # to replicate result
covFunction = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=0.1) # covFunction.plot() # plot of the covariance fn
# Let us sample from the GP
# A psd-matrix can be seen as the covariance of a Gaussian vector. For example, we
# can simulate sample paths from a GP as follow:
xTest = np.linspace(0., 1. , 100) # 100 points evenly spaced over [0,1]
xTest = xTest[:, None] # same as xTrain.shape = (len(xTrain), 1)
mean = np.zeros((100)) # vector of the means
covMatrix = covFunction.K(xTest, xTest) # covariance matrix
# Generate 20 sample path with mean mu and covariance C
Z = np.random.multivariate_normal(mean, covMatrix, 1) # just take 1 sample, but can change that to more
hFig = plt.figure("100D Gaussian")
plt.plot(Z.T, "k*", ms=8) # marker size
plt.xlabel("index ($i$)", fontsize=20)
plt.ylabel("$\mathbf{x}_i$", fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.ylim(-2.6, 2.6)
# Save figure
fileName = "Gaussian100D.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure

#----------------------------------------------------------------------------------------------------------------------#
# Sample from a Infinity-dimensional Gaussian
#----------------------------------------------------------------------------------------------------------------------#
np.random.seed(120) # to replicate result
covFunction = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=0.01) # covFunction.plot() # plot of the covariance fn
# Let us sample from the GP
# A psd-matrix can be seen as the covariance of a Gaussian vector. For example, we
# can simulate sample paths from a GP as follow:
xTest = np.linspace(0., 1. , 1000) # 100 points evenly spaced over [0,1]
xTest = xTest[:, None] # same as xTrain.shape = (len(xTrain), 1)
mean = np.zeros((1000)) # vector of the means
covMatrix = covFunction.K(xTest, xTest) # covariance matrix
# Generate 20 sample path with mean mu and covariance C
Z = np.random.multivariate_normal(mean, covMatrix, 1) # just take 1 sample, but can change that to more
hFig = plt.figure("100D Gaussian")
plt.plot(Z.T, "k", ms=8) # marker size
plt.xlabel("index ($i$)", fontsize=20)
plt.ylabel("$f_i$", fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.ylim(-2.6, 2.6)
# Save figure
fileName = "GaussianInfD.pdf"
with PdfPages(RESULTS_LOC + fileName) as pdf:     
        pdf.savefig(hFig) # save figure
        plt.close(hFig) # close figure