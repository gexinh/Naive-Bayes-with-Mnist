function [likelihood_matrix, priors, evidences] = naive_bayes_train(training_vectors, training_classes, n_classes, k)
    % we denote the symbol as follow:
    % N-feature_space;K-category;M-sample_size
    likelihood_matrix = zeros(n_classes, size(training_vectors,2));  %    
    priors = zeros(n_classes, 1);
    evidences = zeros(size(training_vectors,2), 1);
   
    for class=1:n_classes
       fm = training_vectors(find(training_classes == (class-1)), :);

        % calc and store likelihoods
        part_1=sum(fm,1) + k;
        part_2=size(fm,1) + k * size(training_vectors,2);
        likelihoods = (sum(fm,1) + k) ./ (size(fm,1) + k * size(training_vectors,2)); % laplacan smoothing  
        likelihood_matrix(class, :) = likelihoods;

        % calc and store priors
        priors(class) = (size(fm,1) + k) / (size(training_vectors,1) + k*n_classes); % laplactian smoothing
    end;

    % calc evidences
    evidences = ( (sum(training_vectors,1)+k) ./ (size(training_vectors,1)+k*2) )'; % laplacian smoothing
    %evidences = (priors'*likelihood_matrix)';
end