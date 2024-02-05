import random
import numpy as np
import math
import scipy.special as sp
from scipy.stats import rayleigh, lognorm, beta, chi2, rice #we use this because we know they are correct
import matplotlib.pyplot as plt

#------------------------------------------------------------WARNING---------------------------------------------------------------
#This code generates plots full screen, to close them press 'q' or comment the 2nd 3rd to last lines
#----------------------------------------------------------------------------------------------------------------------------------

class Poisson:#my implementation of the poisson distribution, used for the Rice distribution
    def __init__(self,l) -> None:
        self.l=l
    
    def compute(self):#use the algorithm on the slides
        n=0
        q=1
        while True:
            u_n=random.uniform(0,1)
            q=q*u_n
            if q<math.exp(-self.l):
                return n
            else:
                n+=1

class Normal:#my implementation of the normal distribution, used for the ChiSquared
    def __init__(self,mu,sigma) -> None:
        self.mu=mu
        self.sigma=sigma
    
    def compute(self,n=12):#compute the value using the algorithm in the slides
        return self.mu+self.sigma*((sum([random.uniform(0,1) for _ in range(n)])-n/2)/math.sqrt(n/12))
    
class Rayleigh:#Rayleigh distribution
    def __init__(self,sigma):
        self.sigma_squared=sigma**2
        self.mean=math.sqrt(self.sigma_squared*math.pi/2)#compute the mean directly
        self.variance=(2-math.pi/2)*self.sigma_squared#compute the variance directly
        
    def pdf(self,points):
        return rayleigh.pdf(points)#return points in the analytical pdf

    def cdf(self,points):
        return rayleigh.cdf(points)#return the points in the analytical cdf

    def compute(self):#generate the random variable using the inverse-transform method
        u=random.uniform(0,1)
        z=math.sqrt(-2*self.sigma_squared*math.log(1-u))
        return z

class LogNormal:#Lognormal distribution
    def __init__(self,mu,sigma_squared):
        self.sigma_squared=sigma_squared
        self.mu=mu
        self.mean=math.exp(self.mu+self.sigma_squared/2)#compute the mean directly
        self.variance=math.exp(2*self.mu+self.sigma_squared)*(math.exp(self.sigma_squared)-1)#compute the variance directly

    def pdf(self,points):
        return lognorm.pdf(points,math.sqrt(self.sigma_squared))#return the points in the analytical pdf
    
    def cdf(self,points):
        return lognorm.cdf(points,math.sqrt(self.sigma_squared))#return the points in the analytical cdf

    def util_pdf(self,x):#my implementation of the pdf
        num=math.exp(-((math.log(x)-self.mu)**2)/(2*self.sigma_squared))
        den=x*math.sqrt(2*math.pi*self.sigma_squared)
        return num/den

    def compute(self):#generate a random variable using the acceptance-rejection technique
        while True:
            x=random.uniform(0,10)
            y=random.uniform(0,4)
            if y<=self.util_pdf(x):
                return x

class Beta:
    def __init__(self,alpha,beta) -> None:
        self.alpha=alpha
        self.beta=beta
        self.mean=self.alpha/(self.alpha+self.beta)#compute the mean directly
        self.variance=(self.alpha*self.beta)/((self.alpha+self.beta)**2*(self.alpha+self.beta+1))#compute the variance directly

    def pdf(self,points):
        return beta.pdf(points, self.alpha, self.beta)#return the points in the analytical pdf
    
    def cdf(self,points):
        return beta.cdf(points, self.alpha, self.beta)#return the points in the analytical cdf
    
    def B(self):#useful to compute the pdf
        return(math.gamma(self.alpha)*math.gamma(self.beta))/math.gamma(self.alpha+self.beta)

    def util_pdf(self,x):#my implementation of the pdf
        num=x**(self.alpha-1)*(1-x)**(self.beta-1)
        return 1/self.B()*num
    
    def compute(self):#generate a random variable using the acceptance-rejection technique
        while True:
            x=random.uniform(0,1)
            y=random.uniform(0,2.5)
            if y<=self.util_pdf(x):
                return x

class ChiSquare:
    def __init__(self, dof, mu=0, sigma=1) -> None:
        self.dof=dof
        self.mu=mu
        self.sigma=sigma
        self.mean=dof#compute the mean directly
        self.variance=2*dof#compute the variance directly

    def pdf(self,points):
        return chi2.pdf(points, self.dof)#return the points in the analytical pdf
    
    def cdf(self,points):
        return chi2.cdf(points, self.dof)#return the points in the analytical cdf
    
    def compute(self):#generate the random variable using the convolution method
        x=0
        for _ in range(self.dof):
            x+=Normal(self.mu,self.sigma).compute()**2
        return x

class Rice:
    def __init__(self, nu, sigma) -> None:
        self.nu=nu
        self.sigma_squared=sigma**2
        self.mean=math.sqrt(self.sigma_squared*math.pi/2)*self.L(-self.nu**2/(2*self.sigma_squared))#compute the mean directly
        self.variance=2*self.sigma_squared+ self.nu**2-(math.pi*self.sigma_squared)*self.L(-self.nu**2/(2*self.sigma_squared))**2/2#compute the variance directly

    def L(self,x):#useful to compute the mean and the variance
        return sp.hyp1f1(-1/2,1,x)
    
    def pdf(self,points):#return the points in the analytical pdf
        return rice.pdf(points, self.nu)
    
    def cdf(self,points):#return the points in the analytical cdf
        return rice.cdf(points, self.nu)

    def compute(self):#generate the random variable using the convolution method
        P=Poisson(self.nu**2/(2*self.sigma_squared)).compute()
        X=ChiSquare(2*P+2).compute()#reuse the already implemented chisquare distribution
        return math.sqrt(self.sigma_squared*X)

if __name__=="__main__":
    #rv=Rayleigh(1)
    #rv=LogNormal(0,0.1)
    #rv=Beta(1,1)
    rv=ChiSquare(3)
    #rv=Rice(4,1)
    for N in [1_000,10_000,100_000]:
        values=[]
        for i in range(N):#generate the random values
            values.append(rv.compute())
        sample_mean=np.average(values)#compute the sample mean
        arithmetic_mean=rv.mean#get the arithmetic mean
        print("--------------------------------------------")
        print(f"using{N=}")
        print(f"{sample_mean=}")
        print(f"{arithmetic_mean=}")
        sample_variance=np.std(values, ddof=1)**2#compute the sample variance
        arithmetic_variance=rv.variance#get the arithmetic variance
        print(f"{sample_variance=}")
        print(f"{arithmetic_variance=}")
        print(f"error on the mean:{abs(arithmetic_mean-sample_mean)/arithmetic_mean*100} %")#compute the relative error on the mean
        print(f"error in the variance:{abs(arithmetic_variance-sample_variance)/arithmetic_variance*100} %")#compute the relative error on the variance

        #Plot the results
        n_bins=100
        x_pdf=np.linspace(min(values),max(values),num=N)#points for the pdf
        x_cdf=np.linspace(min(values),max(values),num=n_bins)#points for the cdf
        pdf=rv.pdf(x_pdf)#get points to sketch the pdf
        cdf=rv.cdf(x_cdf)#get points to sketch the cdf
        counts, bins = np.histogram(values,bins=n_bins)#compute the value of the histogram and the number of bins
        empirical_pdf=counts/(sum(counts)*np.diff(bins))#divide the "hight" of the histogram by the largest area of a rectangle to get the pdf
        empirical_cdf=np.cumsum(counts/N)#emulate the integral to compute the cdf
        plt.subplot(1,2,1)
        plt.title(f"PDF, Number of samples={N}")
        plt.stairs(empirical_pdf,bins,label="Empirical PDF")
        plt.plot(x_pdf,pdf, label="Analytical PDF")
        plt.grid()
        plt.legend()
        plt.subplot(1,2,2)
        plt.title(f"CDF, Number of samples={N}")
        plt.plot(x_cdf,empirical_cdf,"b",label="Empirical CDF")
        plt.plot(x_cdf,cdf,'r--',label="Analytical CDF")
        plt.grid()
        plt.legend()
        #make the images full screen
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        plt.show()