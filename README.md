# CavityFlow_FDM
Cavity flow solver using Finite Differences Method and fractional steps method. 

Fractional steps scheme: 

1. ADVECTIVE TERM
This part is solved using a forward Euler explicit scheme. This part of the code can be modified in the 
main part selecting the precision of the first derivative calculation and the form in which the non linear term is treated. 

2. PRESSURE CALCULATION
The pressure calculation is based in the laplacian of the intermediate velocity field that was calculated in the previous step. For this purpose there are two options: a. Tie the pressure to a known value or b. make the regularization of the term by affecting the divergence of the intermediate velocity field with the matrix q q(T) that is constructed based in the singular vector associated with the minimum singular value of the pressure matrix, calculated before the time loop. 

3. DIFFUSIVE OR VISCOUS TERM
This term is calculated via an implicit Euler scheme and the precision can be selected at the top of the code. 
