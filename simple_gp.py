import random as rn
import numpy as np
import pandas as pd

import matplotlib
#fix WSAgg problem
matplotlib.use('QT4Agg')

from matplotlib import pyplot as plt

def sigma_gp(x1, x2, l=1.0):
   sig = pd.DataFrame(0.0, index=range(len(x1)), columns=range(len(x2)))
   for i in range(len(x1)):
      for j in range(len(x2)):
         sig.at[i,j] = float(np.exp( -0.5 * ( abs(x1[i] - x2[j]) / l )**2))
   return (sig)


def d_range(b,e,s):
   res = []
   while b <=e:
      res.append(round(b,5))
      b = b + s
   return res

def main():

   # define some boundries
   x_min = -5
   x_max = 15
   y_min = -5
   y_max = 5

   # highly densed x_line
   x_star =  d_range(x_min+1,x_max-1,0.2)
   # values we are trying to compute
   # x's from the range
   x_val =  [i for i in range(x_min,x_max)] # [-4,-3,-1,0,2]
   # ys: n random integer, where n i len(x)
   y_val = map(  lambda x : rn.randrange(-10,8)
              , [i for i in range(len(x_val))] ) # [-2,0,1,2,-1]

   # data frame for x,y values
   df_xy = pd.DataFrame( zip(x_val, y_val), columns = ['x','y'] )


   k_xx = sigma_gp(x_val,x_val)
   k_xxs = sigma_gp(x_val,x_star)
   k_xsx = sigma_gp(x_star,x_val)
   k_xsxs = sigma_gp(x_star,x_star)

   b=pd.DataFrame(np.eye(len(k_xx)))
   t_res = np.linalg.lstsq(k_xx,b)

   t_res_bar = np.dot(k_xsx,t_res[0])
   f_star_bar = np.dot(t_res_bar,y_val)

   t_cov_res = np.dot(t_res_bar,k_xxs)
   cov_f_star = k_xsxs - t_cov_res

   n_res = 50
   res_df = pd.DataFrame(0.0, index=range(len(f_star_bar)), columns=range(n_res))


   for i in range(n_res):
      res_df[i,]= np.random.multivariate_normal(f_star_bar, cov_f_star)

   for i in range(n_res):
      plt.plot(range(len(f_star_bar)),res_df.ix[:,i])

   plt.show()
#run main
main()
