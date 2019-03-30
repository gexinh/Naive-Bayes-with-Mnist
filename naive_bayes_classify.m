% Naive Bayes Classifier
% 3. Classification
% Jeffrey Jedele, 2011

function [predicted_classes, posteriors] = naive_bayes_classify(vectors, priors, likelihoods, evidence)
   % N-feature_space;K-category;M-sample_size
    n_classes = size(priors, 1);      %K*1
    n_vectors = size(vectors, 1);     %M*1
    predicted_classes = zeros(n_vectors, 1);     %M*1
    posteriors = zeros(n_vectors, n_classes);    %M*K

    for i=1:n_vectors
        %vectors:M*N
        vector_1 = find(vectors(i, :)' == 1);      % find the N-th features whose value is 1. 
        likelihood_v_1 = likelihoods(:, vector_1);     %get the N-th likelihood probabbility
        vector_0 = find(vectors(i,:)'==0);         %find the N-th features whose value is 0.     
        likelihood_v_0=1-likelihoods(:,vector_0);
        %if we use Log-likelihood form to caculate the likelihood ,then the posterior can be expressed as follow:  
        
        %    posterior=x*log[likelihood_v_1]+(1-x)*log[(1-likelihood_v_1)]+prior
        
        %it looks like the cross entropy form.our object is to optimize this loss function.
        posterior =prod(likelihood_v_0,2) .*prod(likelihood_v_1,2) .* priors;  %prod ÊÇÁ¬³Ëº¯Êý    ./ prod(evidence(vector),1)

        [max_val, class] = max(posterior);
        predicted_classes(i) = class-1;
        posteriors(i,:) = posterior';

    end

end
