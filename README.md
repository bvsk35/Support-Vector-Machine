# Support-Vector-Machine
Gaussian and Polynomial Kernel SVM 
<pre>
Author: Sameer Kumar 🐱‍💻
Date: 09/11/2018
Read Me: 
1 - This code is for building a Gaussian Kernel (RBF) Support Vector Machine (SVM) and the optimization 
problem (Quadratic Programming) is solved using python cvxopt optimization toolbox.
Polynomial Kernel SVM can also be build using this code. 
2 - Input Samples variable indicates total no. of points present in both Positive and Negative Classes
Gaussian Standard Deviation is a free variable used in the Gaussian Kernel
Order is a free variable used to control the order of the polynomial used in Polynomial Kernel
Grid Size is controls the number of points to be searched on 1x1 grid to generate decision boundaries
3 - Quadratic Optimization Problem:
        QP: Minimize- 1/2 * X.T * P * X + q.T * X
        ST: G * X <= h and A * X = b
    Matrices were selected based on above descritpion. But for the detailed description go through following link:
    http://cs229.stanford.edu/notes/cs229-notes3.pdf
4 - Solution for the QP problem will not be accurate i.e. lagrange mulitpliers will not be absolute zero, so I have
made values below 1.0e-04 to be zero. For Non Support Vectors lagrange mulitpliers are zero and for Support Vector they are
greater than zero.
5 - Postive Class is also represented by C_1 and Negative Class is also represented C_-1
Note: I have also attached results. 📈👀
</pre>
