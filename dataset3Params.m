function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the 
%exercise where you select the optimal (C, sigma) learning 
%parameters to use for SVM with RBF kernel
%[C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of %C and sigma. You should complete this function to return the 
%optimal C and sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
clen = 0;
Minerror = 100000;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and %  sigma
%  learning parameters found using the cross validation set.
%  You can use svmPredict to predict the labels on the cross 
%  validation set. For example, 
%  predictions = svmPredict(model, Xval);
%  will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of % C and sigma. You should complete this function to return the 
% optimal C and sigma based on a cross-validation set.
% Search over the parameters C and . 
% For both C and sigma, we suggest trying values in 
% multiplicative steps (e.g., 0:01, 0.03, 0.1, 0.3, 1; 3;    10; 
% 30).  Note that you should try all possible pairs of values for  % C and sigma(e.g., C = 0,3
% and  sigma= 0:1). For example, if you try each of the 8 values % listed above for C
% and for sigma2, you would end up training and evaluating (on 
% the cross validation
% set) a total of 82 = 64 dierent models.
Cp = [0.01;0.03; 0.1; 0.3; 1;.3; 10; 30];
sig = [0.01; 0.03; 0.1; 0.3; 1; .3; 10; 30];
clen = length(Cp); 
for i= 1:clen
   cc = Cp(i);
   for j= 1:length(sig)     % sigma and c same length
      ss= sig(j);
   

      %trains an SVM classifier and returns trained model. X is
      %the matrix of training examples

      model = svmTrain(X,y,cc,@(x1, x2) gaussianKernel(x1,x2,ss));
      predictions = svmPredict(model, Xval);
      perror = mean(double(predictions ~= yval));
     %[C, sigma] = EX6PARAMS(X, y, Xval, yval)
     % save the minimum error
     if perror < Minerror
         Minerror = perror;
         C = cc;
         sigma = ss;
         minp = [C sigma];
     end

  end

end

% =========================================================================

end
