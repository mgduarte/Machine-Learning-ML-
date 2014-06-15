function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp = ones(size(theta)); % I added these two lines 
     temp(1) = 0; 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
pred = X*(theta);
 
% fprintf('%f',m);
J=(1/(2* m))*(sum((pred-y).^2));
reg =(lambda/(2 *m)) *( theta' * theta - theta(1)^2);
J+= reg;
sqrErrors =(pred -y)' * X;
grad =(1/m)*(sqrErrors) + (lambda * (theta .*temp)/ m)'; 
%grad =((1/m)*X'*(sum((pred-y)))' + lambda * (theta .*temp)/ m; 













% =========================================================================
