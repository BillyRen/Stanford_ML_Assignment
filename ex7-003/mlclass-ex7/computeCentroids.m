function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
sum = zeros(K, n);
num = zeros(K, 1);
for iter = 1:K
   for iter2 = 1:m
       if idx(iter2) == iter
           sum(iter,:) = sum(iter,:) + X(iter2,:);
           num(iter) = num(iter) + 1;
       end
       %centroids(iter,:) = sum(iter,:) / num(iter);
   end
   centroids(iter,:) = sum(iter,:) / num(iter);
end
% =============================================================


end

