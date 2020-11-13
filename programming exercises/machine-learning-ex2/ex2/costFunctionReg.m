function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
Jreg = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X * theta;
z = exp(-1.*z);
z = z.+1;
z = 1./z;
t = ones(size(y));
p = ones(size(z));
J = -1/m*(y'*log(z) + (t.-y)'*log(p-z));

for i = 2:size(theta)(1)
 Jreg = Jreg + theta(i)*theta(i);

Jreg = lambda/(2*m)* + Jreg
J = J + Jreg

f = z - y;
for i = 1:size(theta)(1)
   grad(i) = X(:,i)' * f
   end
   
grad = grad./m



% =============================================================

end
