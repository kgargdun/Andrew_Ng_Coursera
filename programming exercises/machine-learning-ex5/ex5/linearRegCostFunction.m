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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta;
t = (h - y).^2;
c1 = sum(t);
t2 = theta(2:end);
t3 = t2.^2;
c2 = lambda* sum(t3);
Jt =  c1 + c2;
Jt = (1/(2*m))*Jt;
J = J + Jt

z = theta;
z(1) = 0;
t = h - y;
gt = t .* X;
gtt = sum(gt)/m;
gtt = gtt(:);
gtt = gtt + (z.*lambda)./m;
grad = gtt;

% =========================================================================
grad = grad(:);
end
