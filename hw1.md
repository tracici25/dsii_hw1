DSII HW1
================

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double()
    ## )

    ## See spec(...) for full column specifications.

# Question 1: Linear model

``` r
set.seed(1)

lm_fit = lm(solubility ~., data = train)
summary(lm_fit)
```

    ## 
    ## Call:
    ## lm(formula = solubility ~ ., data = train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -1.75620 -0.28304  0.01165  0.30030  1.54887 
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)         2.431e+00  2.162e+00   1.124 0.261303    
    ## fp001               3.594e-01  3.185e-01   1.128 0.259635    
    ## fp002               1.456e-01  2.637e-01   0.552 0.580960    
    ## fp003              -3.969e-02  1.314e-01  -0.302 0.762617    
    ## fp004              -3.049e-01  1.371e-01  -2.223 0.026520 *  
    ## fp005               2.837e+00  9.598e-01   2.956 0.003223 ** 
    ## fp006              -6.886e-02  2.041e-01  -0.337 0.735917    
    ## fp007               4.044e-02  1.152e-01   0.351 0.725643    
    ## fp008               1.121e-01  1.636e-01   0.685 0.493331    
    ## fp009              -8.242e-01  8.395e-01  -0.982 0.326536    
    ## fp010               4.193e-01  3.136e-01   1.337 0.181579    
    ## fp011               5.158e-02  2.198e-01   0.235 0.814503    
    ## fp012              -1.346e-02  1.611e-01  -0.084 0.933452    
    ## fp013              -4.519e-01  5.473e-01  -0.826 0.409311    
    ## fp014               3.281e-01  4.550e-01   0.721 0.471044    
    ## fp015              -1.839e-01  1.521e-01  -1.209 0.226971    
    ## fp016              -1.367e-01  1.548e-01  -0.883 0.377340    
    ## fp017              -1.704e-01  1.386e-01  -1.230 0.219187    
    ## fp018              -3.824e-01  2.388e-01  -1.602 0.109655    
    ## fp019              -3.131e-01  3.863e-01  -0.811 0.417862    
    ## fp020               2.072e-01  2.135e-01   0.971 0.332078    
    ## fp021              -5.956e-02  2.632e-01  -0.226 0.821060    
    ## fp022               2.336e-01  3.456e-01   0.676 0.499180    
    ## fp023              -3.193e-01  1.909e-01  -1.672 0.094866 .  
    ## fp024              -4.272e-01  2.827e-01  -1.511 0.131162    
    ## fp025               4.376e-01  4.538e-01   0.964 0.335184    
    ## fp026               2.068e-01  2.564e-01   0.806 0.420273    
    ## fp027               2.424e-01  2.429e-01   0.998 0.318594    
    ## fp028               1.070e-01  1.200e-01   0.892 0.372547    
    ## fp029              -9.857e-02  2.199e-01  -0.448 0.654163    
    ## fp030              -2.361e-01  2.468e-01  -0.957 0.339048    
    ## fp031               8.690e-02  1.346e-01   0.646 0.518754    
    ## fp032              -1.204e+00  7.772e-01  -1.550 0.121628    
    ## fp033               5.766e-01  4.236e-01   1.361 0.173882    
    ## fp034              -1.794e-01  2.618e-01  -0.685 0.493486    
    ## fp035              -2.140e-01  1.704e-01  -1.256 0.209605    
    ## fp036               7.701e-02  1.657e-01   0.465 0.642133    
    ## fp037               1.098e-01  1.725e-01   0.636 0.524693    
    ## fp038               2.721e-01  1.888e-01   1.441 0.150030    
    ## fp039               2.011e-02  2.888e-01   0.070 0.944491    
    ## fp040               5.477e-01  1.890e-01   2.898 0.003873 ** 
    ## fp041              -4.265e-01  3.004e-01  -1.420 0.156143    
    ## fp042              -9.901e-01  7.078e-01  -1.399 0.162294    
    ## fp043              -3.725e-02  2.096e-01  -0.178 0.859011    
    ## fp044              -3.860e-01  2.184e-01  -1.768 0.077562 .  
    ## fp045               2.120e-01  1.299e-01   1.631 0.103238    
    ## fp046              -3.504e-02  2.733e-01  -0.128 0.898010    
    ## fp047              -1.675e-02  1.414e-01  -0.118 0.905775    
    ## fp048               2.610e-01  2.434e-01   1.073 0.283810    
    ## fp049               1.241e-01  1.971e-01   0.630 0.529036    
    ## fp050               9.087e-03  1.410e-01   0.064 0.948648    
    ## fp051               1.050e-01  2.014e-01   0.521 0.602210    
    ## fp052              -4.569e-01  2.482e-01  -1.841 0.066029 .  
    ## fp053               2.994e-01  2.466e-01   1.214 0.225129    
    ## fp054               2.734e-02  1.829e-01   0.149 0.881229    
    ## fp055              -3.662e-01  1.970e-01  -1.858 0.063530 .  
    ## fp056              -2.961e-01  2.979e-01  -0.994 0.320541    
    ## fp057              -1.002e-01  1.379e-01  -0.727 0.467703    
    ## fp058               3.100e-01  8.074e-01   0.384 0.701129    
    ## fp059              -1.615e-01  1.690e-01  -0.956 0.339514    
    ## fp060               2.350e-01  1.474e-01   1.595 0.111209    
    ## fp061              -6.365e-01  1.440e-01  -4.421 1.13e-05 ***
    ## fp062              -5.224e-01  2.961e-01  -1.764 0.078078 .  
    ## fp063              -2.001e+00  1.287e+00  -1.554 0.120553    
    ## fp064               2.549e-01  1.221e-01   2.087 0.037207 *  
    ## fp065              -2.844e-01  1.197e-01  -2.377 0.017714 *  
    ## fp066               2.093e-01  1.264e-01   1.655 0.098301 .  
    ## fp067              -1.406e-01  1.540e-01  -0.913 0.361631    
    ## fp068               4.964e-01  2.028e-01   2.447 0.014630 *  
    ## fp069               1.324e-01  8.824e-02   1.501 0.133885    
    ## fp070               3.453e-03  8.088e-02   0.043 0.965963    
    ## fp071               1.474e-01  1.237e-01   1.192 0.233775    
    ## fp072              -9.773e-01  2.763e-01  -3.537 0.000431 ***
    ## fp073              -4.671e-01  2.072e-01  -2.254 0.024474 *  
    ## fp074               1.793e-01  1.206e-01   1.487 0.137566    
    ## fp075               1.231e-01  1.035e-01   1.188 0.235034    
    ## fp076               5.166e-01  1.704e-01   3.031 0.002525 ** 
    ## fp077               1.644e-01  1.236e-01   1.331 0.183739    
    ## fp078              -3.715e-01  1.588e-01  -2.339 0.019608 *  
    ## fp079               4.254e-01  1.881e-01   2.262 0.023992 *  
    ## fp080               3.101e-01  1.554e-01   1.996 0.046340 *  
    ## fp081              -3.208e-01  1.117e-01  -2.873 0.004192 ** 
    ## fp082               1.243e-01  9.524e-02   1.305 0.192379    
    ## fp083              -6.916e-01  2.134e-01  -3.241 0.001248 ** 
    ## fp084               3.626e-01  2.381e-01   1.523 0.128171    
    ## fp085              -3.310e-01  1.428e-01  -2.317 0.020785 *  
    ## fp086               1.169e-02  9.774e-02   0.120 0.904834    
    ## fp087               4.559e-02  2.797e-01   0.163 0.870568    
    ## fp088               2.416e-01  9.959e-02   2.425 0.015534 *  
    ## fp089               5.999e-01  2.320e-01   2.586 0.009915 ** 
    ## fp090              -2.450e-02  1.154e-01  -0.212 0.831930    
    ## fp091              -2.858e-01  3.185e-01  -0.897 0.369847    
    ## fp092               2.665e-01  2.069e-01   1.288 0.198156    
    ## fp093               1.974e-01  1.087e-01   1.816 0.069803 .  
    ## fp094              -1.991e-01  1.441e-01  -1.381 0.167707    
    ## fp095              -1.403e-01  1.124e-01  -1.248 0.212449    
    ## fp096              -5.024e-01  1.459e-01  -3.445 0.000605 ***
    ## fp097              -2.635e-01  1.666e-01  -1.582 0.114020    
    ## fp098              -2.865e-01  1.633e-01  -1.754 0.079863 .  
    ## fp099               2.592e-01  2.568e-01   1.009 0.313136    
    ## fp100              -4.008e-01  3.034e-01  -1.321 0.186949    
    ## fp101              -1.760e-01  3.019e-01  -0.583 0.560147    
    ## fp102               2.445e-01  3.449e-01   0.709 0.478579    
    ## fp103              -1.493e-01  9.148e-02  -1.632 0.103176    
    ## fp104              -1.428e-01  1.176e-01  -1.214 0.225238    
    ## fp105              -6.912e-02  1.395e-01  -0.495 0.620482    
    ## fp106               1.128e-01  1.288e-01   0.876 0.381495    
    ## fp107               2.778e+00  8.247e-01   3.369 0.000796 ***
    ## fp108               8.836e-03  1.852e-01   0.048 0.961970    
    ## fp109               8.200e-01  2.267e-01   3.617 0.000319 ***
    ## fp110               3.680e-01  3.311e-01   1.111 0.266811    
    ## fp111              -5.565e-01  1.420e-01  -3.918 9.80e-05 ***
    ## fp112              -1.079e-01  2.705e-01  -0.399 0.690108    
    ## fp113               1.511e-01  9.481e-02   1.594 0.111478    
    ## fp114              -1.201e-01  1.891e-01  -0.635 0.525628    
    ## fp115              -1.896e-01  1.405e-01  -1.349 0.177736    
    ## fp116               7.778e-03  1.897e-01   0.041 0.967300    
    ## fp117               2.583e-01  1.779e-01   1.452 0.147070    
    ## fp118              -1.964e-01  1.230e-01  -1.596 0.110940    
    ## fp119               7.515e-01  2.630e-01   2.857 0.004402 ** 
    ## fp120              -1.814e-01  1.794e-01  -1.011 0.312362    
    ## fp121              -4.731e-02  3.957e-01  -0.120 0.904866    
    ## fp122               1.048e-01  1.041e-01   1.007 0.314268    
    ## fp123               3.926e-02  1.765e-01   0.222 0.824066    
    ## fp124               1.235e-01  1.705e-01   0.724 0.469243    
    ## fp125              -2.633e-04  1.151e-01  -0.002 0.998175    
    ## fp126              -2.782e-01  1.177e-01  -2.363 0.018373 *  
    ## fp127              -6.123e-01  1.739e-01  -3.521 0.000457 ***
    ## fp128              -5.424e-01  1.932e-01  -2.807 0.005136 ** 
    ## fp129              -6.731e-02  2.243e-01  -0.300 0.764167    
    ## fp130              -1.034e+00  4.106e-01  -2.518 0.012009 *  
    ## fp131               2.158e-01  1.617e-01   1.335 0.182405    
    ## fp132              -1.976e-01  2.382e-01  -0.830 0.406998    
    ## fp133              -1.573e-01  1.217e-01  -1.293 0.196319    
    ## fp134               2.496e+00  1.196e+00   2.086 0.037310 *  
    ## fp135               1.818e-01  1.319e-01   1.379 0.168460    
    ## fp136              -7.763e-02  3.131e-01  -0.248 0.804237    
    ## fp137              -4.613e-02  2.978e-01  -0.155 0.876947    
    ## fp138              -9.392e-02  1.906e-01  -0.493 0.622251    
    ## fp139               7.659e-02  4.063e-01   0.189 0.850517    
    ## fp140               3.145e-01  2.149e-01   1.463 0.143784    
    ## fp141               2.219e-01  2.765e-01   0.802 0.422532    
    ## fp142               6.272e-01  1.488e-01   4.214 2.83e-05 ***
    ## fp143               9.981e-01  2.929e-01   3.407 0.000692 ***
    ## fp144               2.207e-01  2.839e-01   0.777 0.437195    
    ## fp145              -1.146e-01  1.188e-01  -0.964 0.335169    
    ## fp146              -2.324e-01  2.086e-01  -1.114 0.265716    
    ## fp147               1.502e-01  1.228e-01   1.223 0.221703    
    ## fp148              -1.600e-01  1.319e-01  -1.213 0.225560    
    ## fp149               1.172e-01  1.650e-01   0.710 0.477770    
    ## fp150               9.046e-02  1.577e-01   0.574 0.566368    
    ## fp151               2.899e-01  3.120e-01   0.929 0.353202    
    ## fp152              -2.544e-01  2.990e-01  -0.851 0.395087    
    ## fp153              -3.765e-01  2.773e-01  -1.358 0.175029    
    ## fp154              -1.027e+00  2.033e-01  -5.054 5.50e-07 ***
    ## fp155               4.888e-01  2.916e-01   1.676 0.094163 .  
    ## fp156              -3.602e-02  3.636e-01  -0.099 0.921109    
    ## fp157              -4.715e-01  2.468e-01  -1.910 0.056505 .  
    ## fp158               1.669e-02  1.925e-01   0.087 0.930943    
    ## fp159               1.800e-01  2.432e-01   0.740 0.459378    
    ## fp160               1.525e-02  2.177e-01   0.070 0.944155    
    ## fp161              -2.440e-01  1.433e-01  -1.703 0.089063 .  
    ## fp162               4.910e-02  1.859e-01   0.264 0.791710    
    ## fp163               4.785e-01  3.121e-01   1.533 0.125659    
    ## fp164               5.096e-01  1.899e-01   2.684 0.007446 ** 
    ## fp165               5.793e-01  2.146e-01   2.700 0.007103 ** 
    ## fp166              -6.582e-02  2.185e-01  -0.301 0.763293    
    ## fp167              -6.044e-01  2.515e-01  -2.403 0.016502 *  
    ## fp168              -1.187e-01  1.872e-01  -0.634 0.526173    
    ## fp169              -1.705e-01  8.312e-02  -2.051 0.040650 *  
    ## fp170              -7.902e-02  1.560e-01  -0.506 0.612745    
    ## fp171               4.651e-01  1.186e-01   3.922 9.64e-05 ***
    ## fp172              -4.426e-01  2.440e-01  -1.814 0.070120 .  
    ## fp173               4.243e-01  1.657e-01   2.561 0.010634 *  
    ## fp174              -1.010e-01  2.098e-01  -0.481 0.630311    
    ## fp175              -4.657e-02  2.481e-01  -0.188 0.851136    
    ## fp176               9.736e-01  2.644e-01   3.682 0.000249 ***
    ## fp177               1.386e-01  2.393e-01   0.579 0.562538    
    ## fp178               6.497e-02  2.079e-01   0.313 0.754691    
    ## fp179              -3.415e-02  2.232e-01  -0.153 0.878437    
    ## fp180              -7.905e-01  5.523e-01  -1.431 0.152839    
    ## fp181               4.925e-01  3.218e-01   1.531 0.126309    
    ## fp182              -1.124e-01  1.310e-01  -0.858 0.391384    
    ## fp183               2.998e-01  7.143e-01   0.420 0.674836    
    ## fp184               4.876e-01  1.580e-01   3.087 0.002103 ** 
    ## fp185              -3.778e-01  2.037e-01  -1.854 0.064108 .  
    ## fp186              -3.654e-01  1.953e-01  -1.871 0.061710 .  
    ## fp187               4.457e-01  2.682e-01   1.662 0.097015 .  
    ## fp188               1.475e-01  1.258e-01   1.172 0.241519    
    ## fp189              -1.984e-02  3.468e-01  -0.057 0.954384    
    ## fp190               2.629e-01  3.018e-01   0.871 0.383981    
    ## fp191               2.799e-01  1.465e-01   1.911 0.056388 .  
    ## fp192              -2.404e-01  2.751e-01  -0.874 0.382534    
    ## fp193               1.502e-01  1.494e-01   1.005 0.315159    
    ## fp194               8.029e-01  6.379e-01   1.259 0.208566    
    ## fp195               5.967e-02  3.435e-01   0.174 0.862158    
    ## fp196               1.091e-02  2.544e-01   0.043 0.965812    
    ## fp197              -3.736e-02  1.569e-01  -0.238 0.811793    
    ## fp198               1.896e-01  2.665e-01   0.712 0.476893    
    ## fp199              -9.932e-02  1.797e-01  -0.553 0.580702    
    ## fp200              -6.421e-02  2.161e-01  -0.297 0.766462    
    ## fp201              -4.838e-01  1.980e-01  -2.444 0.014771 *  
    ## fp202               5.664e-01  1.869e-01   3.031 0.002527 ** 
    ## fp203               2.586e-01  6.447e-01   0.401 0.688462    
    ## fp204              -1.371e-01  2.543e-01  -0.539 0.590008    
    ## fp205               7.177e-02  1.561e-01   0.460 0.645857    
    ## fp206              -6.769e-02  1.860e-01  -0.364 0.716094    
    ## fp207              -5.538e-03  2.060e-01  -0.027 0.978560    
    ## fp208              -5.338e-01  6.324e-01  -0.844 0.398925    
    ## mol_weight         -1.232e+00  2.296e-01  -5.365 1.09e-07 ***
    ## num_atoms          -1.478e+01  3.473e+00  -4.257 2.35e-05 ***
    ## num_non_h_atoms     1.795e+01  3.166e+00   5.670 2.07e-08 ***
    ## num_bonds           9.843e+00  2.681e+00   3.671 0.000260 ***
    ## num_non_h_bonds    -1.030e+01  1.793e+00  -5.746 1.35e-08 ***
    ## num_mult_bonds      2.107e-01  1.754e-01   1.201 0.229990    
    ## num_rot_bonds      -5.213e-01  1.334e-01  -3.908 0.000102 ***
    ## num_dbl_bonds      -7.492e-01  3.163e-01  -2.369 0.018111 *  
    ## num_aromatic_bonds -2.364e+00  6.232e-01  -3.794 0.000161 ***
    ## num_hydrogen        8.347e-01  1.880e-01   4.439 1.04e-05 ***
    ## num_carbon          1.730e-02  3.763e-01   0.046 0.963335    
    ## num_nitrogen        6.125e+00  3.045e+00   2.011 0.044645 *  
    ## num_oxygen          2.389e+00  4.523e-01   5.283 1.69e-07 ***
    ## num_sulfer         -8.508e+00  3.619e+00  -2.351 0.018994 *  
    ## num_chlorine       -7.449e+00  1.989e+00  -3.744 0.000195 ***
    ## num_halogen         1.408e+00  2.109e+00   0.668 0.504615    
    ## num_rings           1.276e+00  6.716e-01   1.901 0.057731 .  
    ## hydrophilic_factor  1.099e-02  1.137e-01   0.097 0.922998    
    ## surface_area1       8.825e-02  6.058e-02   1.457 0.145643    
    ## surface_area2       9.555e-02  5.615e-02   1.702 0.089208 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.5524 on 722 degrees of freedom
    ## Multiple R-squared:  0.9446, Adjusted R-squared:  0.9271 
    ## F-statistic: 54.03 on 228 and 722 DF,  p-value: < 2.2e-16

``` r
final_lm = predict(lm_fit, newdata = test)
mse_lm = mean((final_lm - test$solubility)^2)
mse_lm
```

    ## [1] 0.5558898

# Question 2: Ridge regression model

``` r
set.seed(1)

# Cross Validation
cv.ridge = cv.glmnet(xtrain, ytrain, 
                     type.measure = "mse",
                     alpha = 0,
                     lambda = exp(seq(-10, 5, length = 1000)))
plot(cv.ridge)
```

![](hw1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
cv.ridge$lambda.min # min MSE
```

    ## [1] 0.06504131

``` r
cv.ridge$lambda.1se # 1se
```

    ## [1] 0.2588902

``` r
ridge = glmnet(xtrain, ytrain, 
               standardize = TRUE,
               alpha = 0,
               lambda = cv.ridge$lambda.min)
final_ridge = predict(ridge, newx = model.matrix(solubility ~., test)[,-1], s = cv.ridge$lambda.min, type = "response")
mse_ridge = mean((final_ridge - test$solubility)^2)
mse_ridge
```

    ## [1] 0.5133069

# Question 3: Lasso Model

``` r
set.seed(1)

cv.lasso = cv.glmnet(xtrain, ytrain, 
                     type.measure = "mse",
                     alpha = 1,
                     lambda = exp(seq(-10, 3, length = 1000)))
plot(cv.lasso)
```

![](hw1_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
cv.lasso$lambda.min # min MSE
```

    ## [1] 0.0046664

``` r
cv.lasso$lambda.1se # 1se
```

    ## [1] 0.02111318

``` r
lasso = glmnet(xtrain, ytrain, 
               standardize = TRUE,
               alpha = 1,
               lambda = cv.lasso$lambda.min)
final_lasso = predict(lasso, newx = model.matrix(solubility ~., test)[,-1], s = cv.lasso$lambda.min, type = "response")
mse_lasso = mean((final_lasso - test$solubility)^2)
mse_lasso
```

    ## [1] 0.4950328

``` r
sum(final_lasso!=0) # non-zero coefficients
```

    ## [1] 316

# Question 4: Principle Component Regression Model

``` r
set.seed(1)
pcr = pcr(solubility ~., data = train, scale = TRUE, validation = "CV")
summary(pcr) # n of components = 228
```

    ## Data:    X dimension: 951 228 
    ##  Y dimension: 951 1
    ## Fit method: svdpc
    ## Number of components considered: 228
    ## 
    ## VALIDATION: RMSEP
    ## Cross-validated using 10 random segments.
    ##        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
    ## CV           2.048    2.044    1.979    1.712    1.605    1.582    1.451
    ## adjCV        2.048    2.044    1.978    1.711    1.602    1.618    1.450
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV       1.301    1.295    1.295     1.277     1.254     1.248     1.243
    ## adjCV    1.297    1.292    1.294     1.276     1.249     1.246     1.243
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps  20 comps
    ## CV        1.198     1.181     1.120     1.062     1.051     1.040     1.019
    ## adjCV     1.197     1.183     1.117     1.054     1.049     1.038     1.012
    ##        21 comps  22 comps  23 comps  24 comps  25 comps  26 comps  27 comps
    ## CV        1.008     1.007    0.9778    0.9785    0.9757    0.9695    0.9683
    ## adjCV     1.006     1.007    0.9742    0.9759    0.9742    0.9661    0.9663
    ##        28 comps  29 comps  30 comps  31 comps  32 comps  33 comps  34 comps
    ## CV       0.9637    0.9661    0.9665    0.9468    0.9310    0.9192    0.9202
    ## adjCV    0.9629    0.9645    0.9673    0.9438    0.9285    0.9159    0.9188
    ##        35 comps  36 comps  37 comps  38 comps  39 comps  40 comps  41 comps
    ## CV       0.9023    0.8882    0.8802    0.8814    0.8828    0.8724    0.8709
    ## adjCV    0.9004    0.8854    0.8768    0.8784    0.8838    0.8688    0.8684
    ##        42 comps  43 comps  44 comps  45 comps  46 comps  47 comps  48 comps
    ## CV       0.8715    0.8657    0.8515    0.8495    0.8519    0.8451    0.8446
    ## adjCV    0.8702    0.8659    0.8491    0.8458    0.8489    0.8420    0.8420
    ##        49 comps  50 comps  51 comps  52 comps  53 comps  54 comps  55 comps
    ## CV       0.8441    0.8436    0.8401    0.8351    0.8326    0.8328    0.8323
    ## adjCV    0.8415    0.8419    0.8386    0.8305    0.8297    0.8295    0.8297
    ##        56 comps  57 comps  58 comps  59 comps  60 comps  61 comps  62 comps
    ## CV       0.8334    0.8304    0.8273    0.8255    0.8219    0.8103    0.8085
    ## adjCV    0.8309    0.8313    0.8295    0.8242    0.8205    0.8027    0.8023
    ##        63 comps  64 comps  65 comps  66 comps  67 comps  68 comps  69 comps
    ## CV       0.8114    0.8103    0.8111    0.8157    0.8104    0.8110    0.8088
    ## adjCV    0.8063    0.8067    0.8076    0.8133    0.8070    0.8084    0.8055
    ##        70 comps  71 comps  72 comps  73 comps  74 comps  75 comps  76 comps
    ## CV       0.8081    0.8069    0.7986    0.7993    0.7972    0.7967    0.7950
    ## adjCV    0.8052    0.8061    0.7922    0.7933    0.7917    0.7922    0.7911
    ##        77 comps  78 comps  79 comps  80 comps  81 comps  82 comps  83 comps
    ## CV       0.7934    0.7936    0.7895    0.7905    0.7903    0.7932    0.7952
    ## adjCV    0.7904    0.7908    0.7855    0.7871    0.7875    0.7881    0.7918
    ##        84 comps  85 comps  86 comps  87 comps  88 comps  89 comps  90 comps
    ## CV       0.7944    0.7932    0.7929    0.7935    0.7934    0.7921    0.7883
    ## adjCV    0.7915    0.7884    0.7885    0.7903    0.7906    0.7902    0.7861
    ##        91 comps  92 comps  93 comps  94 comps  95 comps  96 comps  97 comps
    ## CV       0.7875    0.7865    0.7861    0.7834    0.7812    0.7802    0.7833
    ## adjCV    0.7829    0.7794    0.7790    0.7769    0.7756    0.7749    0.7791
    ##        98 comps  99 comps  100 comps  101 comps  102 comps  103 comps
    ## CV        0.784    0.7850     0.7816     0.7759     0.7758     0.7695
    ## adjCV     0.780    0.7818     0.7751     0.7707     0.7709     0.7660
    ##        104 comps  105 comps  106 comps  107 comps  108 comps  109 comps
    ## CV        0.7658     0.7647     0.7651     0.7661     0.7668     0.7647
    ## adjCV     0.7598     0.7593     0.7603     0.7617     0.7635     0.7572
    ##        110 comps  111 comps  112 comps  113 comps  114 comps  115 comps
    ## CV        0.7665     0.7636     0.7602     0.7627     0.7639     0.7665
    ## adjCV     0.7596     0.7577     0.7554     0.7584     0.7592     0.7626
    ##        116 comps  117 comps  118 comps  119 comps  120 comps  121 comps
    ## CV        0.7657     0.7668     0.7648     0.7624      0.762     0.7597
    ## adjCV     0.7607     0.7626     0.7636     0.7611      0.761     0.7529
    ##        122 comps  123 comps  124 comps  125 comps  126 comps  127 comps
    ## CV        0.7579     0.7572     0.7568     0.7548      0.749     0.7489
    ## adjCV     0.7531     0.7527     0.7521     0.7466      0.742     0.7435
    ##        128 comps  129 comps  130 comps  131 comps  132 comps  133 comps
    ## CV        0.7482     0.7482     0.7433     0.7367     0.7375     0.7374
    ## adjCV     0.7443     0.7429     0.7410     0.7289     0.7282     0.7287
    ##        134 comps  135 comps  136 comps  137 comps  138 comps  139 comps
    ## CV        0.7343     0.7294     0.7282     0.7303     0.7282     0.7280
    ## adjCV     0.7276     0.7247     0.7216     0.7240     0.7224     0.7224
    ##        140 comps  141 comps  142 comps  143 comps  144 comps  145 comps
    ## CV        0.7276     0.7208     0.7206     0.7174     0.7156     0.7121
    ## adjCV     0.7219     0.7139     0.7154     0.7147     0.7059     0.7048
    ##        146 comps  147 comps  148 comps  149 comps  150 comps  151 comps
    ## CV        0.7117     0.7085     0.7082     0.7066     0.7061     0.7081
    ## adjCV     0.7038     0.7014     0.6993     0.6990     0.6980     0.7008
    ##        152 comps  153 comps  154 comps  155 comps  156 comps  157 comps
    ## CV        0.7055     0.7071     0.7106     0.7114     0.7102     0.7106
    ## adjCV     0.6985     0.7001     0.7043     0.7034     0.7030     0.7020
    ##        158 comps  159 comps  160 comps  161 comps  162 comps  163 comps
    ## CV        0.7086     0.7096     0.7104     0.7114     0.7106     0.7125
    ## adjCV     0.7001     0.7018     0.7027     0.7031     0.7029     0.7057
    ##        164 comps  165 comps  166 comps  167 comps  168 comps  169 comps
    ## CV        0.7138     0.7153     0.7162     0.7145     0.7154     0.7147
    ## adjCV     0.7059     0.7078     0.7073     0.7062     0.7072     0.7069
    ##        170 comps  171 comps  172 comps  173 comps  174 comps  175 comps
    ## CV        0.7146     0.7167     0.7188     0.7191     0.7169     0.7143
    ## adjCV     0.7065     0.7085     0.7107     0.7116     0.7085     0.7066
    ##        176 comps  177 comps  178 comps  179 comps  180 comps  181 comps
    ## CV        0.7185     0.7188     0.7156     0.7169     0.7157     0.7161
    ## adjCV     0.7095     0.7096     0.7075     0.7089     0.7081     0.7071
    ##        182 comps  183 comps  184 comps  185 comps  186 comps  187 comps
    ## CV        0.7188     0.7183     0.7182     0.7176     0.7184     0.7211
    ## adjCV     0.7103     0.7088     0.7096     0.7090     0.7083     0.7115
    ##        188 comps  189 comps  190 comps  191 comps  192 comps  193 comps
    ## CV        0.7234     0.7241     0.7248     0.7242     0.7276     0.7298
    ## adjCV     0.7138     0.7146     0.7154     0.7151     0.7186     0.7207
    ##        194 comps  195 comps  196 comps  197 comps  198 comps  199 comps
    ## CV        0.7338     0.7331     0.7312     0.7334     0.7329     0.7384
    ## adjCV     0.7240     0.7236     0.7194     0.7230     0.7227     0.7286
    ##        200 comps  201 comps  202 comps  203 comps  204 comps  205 comps
    ## CV        0.7375     0.7392     0.7345     0.7387     0.7401     0.7355
    ## adjCV     0.7263     0.7283     0.7231     0.7275     0.7290     0.7247
    ##        206 comps  207 comps  208 comps  209 comps  210 comps  211 comps
    ## CV        0.7321     0.7323     0.7268     0.7202     0.7258     0.7274
    ## adjCV     0.7197     0.7210     0.7151     0.7085     0.7142     0.7158
    ##        212 comps  213 comps  214 comps  215 comps  216 comps  217 comps
    ## CV        0.7371     0.7405     0.7448     0.7441     0.7428     0.7414
    ## adjCV     0.7247     0.7280     0.7319     0.7312     0.7299     0.7272
    ##        218 comps  219 comps  220 comps  221 comps  222 comps  223 comps
    ## CV        0.7454     0.7420     0.7433     0.7508     0.7494     0.7489
    ## adjCV     0.7314     0.7286     0.7300     0.7372     0.7360     0.7348
    ##        224 comps  225 comps  226 comps  227 comps  228 comps
    ## CV        0.7468     0.7432     0.7455     0.7434  2.455e+12
    ## adjCV     0.7333     0.7292     0.7314     0.7310  2.329e+12
    ## 
    ## TRAINING: % variance explained
    ##             1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps
    ## X            12.417   23.083    30.29    34.91    39.27    43.53    46.98
    ## solubility    0.734    7.182    30.52    39.36    39.52    50.82    60.83
    ##             8 comps  9 comps  10 comps  11 comps  12 comps  13 comps  14 comps
    ## X             50.08    53.04     55.46     57.67     59.81     61.72     63.43
    ## solubility    61.00    61.01     62.57     64.10     64.17     64.36     67.12
    ##             15 comps  16 comps  17 comps  18 comps  19 comps  20 comps
    ## X              64.82     66.16     67.40     68.58     69.68     70.73
    ## solubility     68.79     71.69     74.75     74.96     75.59     76.72
    ##             21 comps  22 comps  23 comps  24 comps  25 comps  26 comps
    ## X              71.76     72.72     73.64     74.48     75.31     76.09
    ## solubility     76.96     77.00     78.34     78.40     78.48     78.97
    ##             27 comps  28 comps  29 comps  30 comps  31 comps  32 comps
    ## X              76.85     77.57     78.29     78.95     79.59     80.22
    ## solubility     79.03     79.20     79.42     79.43     80.34     80.94
    ##             33 comps  34 comps  35 comps  36 comps  37 comps  38 comps
    ## X              80.81     81.38     81.92     82.46     82.96     83.45
    ## solubility     81.52     81.53     82.15     82.66     82.96     82.97
    ##             39 comps  40 comps  41 comps  42 comps  43 comps  44 comps
    ## X              83.92     84.37     84.82     85.23     85.64     86.03
    ## solubility     82.98     83.49     83.58     83.61     83.73     84.41
    ##             45 comps  46 comps  47 comps  48 comps  49 comps  50 comps
    ## X              86.42     86.78     87.13     87.46     87.77     88.08
    ## solubility     84.54     84.56     84.82     84.82     84.93     84.99
    ##             51 comps  52 comps  53 comps  54 comps  55 comps  56 comps
    ## X              88.39     88.68     88.97     89.25     89.52     89.77
    ## solubility     85.19     85.46     85.48     85.55     85.57     85.61
    ##             57 comps  58 comps  59 comps  60 comps  61 comps  62 comps
    ## X              90.02     90.26     90.51     90.75     90.97     91.19
    ## solubility     85.62     85.70     85.89     86.12     86.61     86.65
    ##             63 comps  64 comps  65 comps  66 comps  67 comps  68 comps
    ## X              91.41     91.62     91.83     92.03     92.23     92.42
    ## solubility     86.66     86.66     86.67     86.69     86.82     86.83
    ##             69 comps  70 comps  71 comps  72 comps  73 comps  74 comps
    ## X              92.60     92.77     92.95     93.12     93.28     93.44
    ## solubility     86.92     86.94     87.00     87.45     87.48     87.50
    ##             75 comps  76 comps  77 comps  78 comps  79 comps  80 comps
    ## X              93.60     93.76     93.91     94.06     94.20     94.34
    ## solubility     87.51     87.55     87.57     87.62     87.74     87.76
    ##             81 comps  82 comps  83 comps  84 comps  85 comps  86 comps
    ## X              94.47     94.61     94.74     94.86     94.99     95.11
    ## solubility     87.83     87.95     87.95     88.00     88.11     88.11
    ##             87 comps  88 comps  89 comps  90 comps  91 comps  92 comps
    ## X              95.22     95.34     95.45     95.56     95.66     95.77
    ## solubility     88.13     88.14     88.14     88.23     88.41     88.60
    ##             93 comps  94 comps  95 comps  96 comps  97 comps  98 comps
    ## X              95.87     95.97     96.07     96.16     96.26     96.35
    ## solubility     88.67     88.68     88.71     88.72     88.72     88.74
    ##             99 comps  100 comps  101 comps  102 comps  103 comps  104 comps
    ## X              96.44      96.53      96.61      96.70      96.78      96.86
    ## solubility     88.74      88.94      88.97      89.02      89.12      89.30
    ##             105 comps  106 comps  107 comps  108 comps  109 comps  110 comps
    ## X               96.94      97.02      97.09      97.17      97.24      97.31
    ## solubility      89.33      89.33      89.34      89.39      89.62      89.64
    ##             111 comps  112 comps  113 comps  114 comps  115 comps  116 comps
    ## X               97.38      97.45      97.51      97.58      97.64      97.70
    ## solubility      89.65      89.66      89.77      89.81      89.81      89.87
    ##             117 comps  118 comps  119 comps  120 comps  121 comps  122 comps
    ## X               97.76      97.82      97.88      97.94      98.00      98.05
    ## solubility      89.88      89.88      90.00      90.06      90.34      90.36
    ##             123 comps  124 comps  125 comps  126 comps  127 comps  128 comps
    ## X               98.11      98.16      98.21      98.26      98.31      98.36
    ## solubility      90.44      90.49      90.67      90.69      90.70      90.70
    ##             129 comps  130 comps  131 comps  132 comps  133 comps  134 comps
    ## X               98.41      98.45      98.50      98.54      98.59      98.63
    ## solubility      90.79      90.79      91.14      91.24      91.25      91.25
    ##             135 comps  136 comps  137 comps  138 comps  139 comps  140 comps
    ## X               98.67      98.71      98.75      98.79      98.82      98.86
    ## solubility      91.25      91.34      91.39      91.41      91.42      91.49
    ##             141 comps  142 comps  143 comps  144 comps  145 comps  146 comps
    ## X               98.89      98.93      98.96      99.00      99.03      99.06
    ## solubility      91.63      91.65      91.65      91.91      91.91      91.96
    ##             147 comps  148 comps  149 comps  150 comps  151 comps  152 comps
    ## X               99.09      99.12      99.15      99.18      99.20      99.23
    ## solubility      91.97      92.06      92.06      92.12      92.12      92.12
    ##             153 comps  154 comps  155 comps  156 comps  157 comps  158 comps
    ## X               99.26      99.28      99.31      99.33      99.35      99.38
    ## solubility      92.15      92.16      92.26      92.26      92.35      92.37
    ##             159 comps  160 comps  161 comps  162 comps  163 comps  164 comps
    ## X               99.40      99.42      99.44      99.46      99.48      99.50
    ## solubility      92.37      92.37      92.40      92.41      92.41      92.47
    ##             165 comps  166 comps  167 comps  168 comps  169 comps  170 comps
    ## X               99.52      99.54      99.56      99.57      99.59      99.61
    ## solubility      92.47      92.53      92.54      92.55      92.55      92.57
    ##             171 comps  172 comps  173 comps  174 comps  175 comps  176 comps
    ## X               99.62      99.64      99.65      99.67      99.68       99.7
    ## solubility      92.58      92.58      92.58      92.64      92.64       92.7
    ##             177 comps  178 comps  179 comps  180 comps  181 comps  182 comps
    ## X               99.71      99.73      99.74      99.75      99.76      99.77
    ## solubility      92.72      92.72      92.76      92.76      92.84      92.84
    ##             183 comps  184 comps  185 comps  186 comps  187 comps  188 comps
    ## X               99.79       99.8      99.81      99.82      99.83      99.84
    ## solubility      92.89       92.9      92.92      92.98      92.98      92.99
    ##             189 comps  190 comps  191 comps  192 comps  193 comps  194 comps
    ## X               99.85      99.86      99.86      99.87      99.88      99.89
    ## solubility      93.00      93.02      93.02      93.02      93.03      93.07
    ##             195 comps  196 comps  197 comps  198 comps  199 comps  200 comps
    ## X               99.90      99.90      99.91      99.92      99.92      99.93
    ## solubility      93.11      93.24      93.24      93.25      93.26      93.35
    ##             201 comps  202 comps  203 comps  204 comps  205 comps  206 comps
    ## X               99.94      99.94      99.95      99.95      99.96      99.96
    ## solubility      93.35      93.42      93.42      93.42      93.48      93.57
    ##             207 comps  208 comps  209 comps  210 comps  211 comps  212 comps
    ## X               99.97      99.97      99.97      99.98      99.98      99.98
    ## solubility      93.57      93.65      93.69      93.70      93.70      93.73
    ##             213 comps  214 comps  215 comps  216 comps  217 comps  218 comps
    ## X               99.99      99.99      99.99      99.99      99.99      99.99
    ## solubility      93.73      93.76      93.78      93.82      93.88      93.90
    ##             219 comps  220 comps  221 comps  222 comps  223 comps  224 comps
    ## X              100.00     100.00     100.00     100.00     100.00     100.00
    ## solubility      93.92      93.93      93.94      93.97      94.04      94.04
    ##             225 comps  226 comps  227 comps  228 comps
    ## X              100.00     100.00     100.00     100.00
    ## solubility      94.16      94.16      94.16      94.46

``` r
validationplot(pcr, val.type = "MSEP")
```

![](hw1_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
final_pcr = predict(pcr, newdata = test, ncomp = 228)
mse_pcr = mean((final_pcr - test$solubility)^2)
mse_pcr
```

    ## [1] 0.5558898

# Question 5

``` r
data.frame(mse_lm, mse_ridge, mse_lasso, mse_pcr)
```

    ##      mse_lm mse_ridge mse_lasso   mse_pcr
    ## 1 0.5558898 0.5133069 0.4950328 0.5558898

Based on the table above, Lasso model has the least MSE, therefore,
Lasso is preferred to predict solubility.