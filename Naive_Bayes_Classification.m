%% load the data
data_set = loadMNISTImages('./train-images-idx3-ubyte');
data_labels=loadMNISTLabels('train-labels-idx1-ubyte');
test_set=loadMNISTImages('t10k-images-idx3-ubyte');
test_labels=loadMNISTLabels('t10k-labels-idx1-ubyte');
%% preprocess
data_set=data_set'>0;      %binary process:大于0的元素作为一个布尔值赋给矩阵
test_set=test_set'>0;
%% split the crossvalidation and test set
training_set=data_set(1:50000,:);
training_labels=data_labels(1:50000,:);

crossval_set=data_set(50001:60000,:);
crossval_labels=data_labels(50001:60000,:);

%% output 10 cross validation images to test
images_1=reshape(crossval_set(1:10,:),10,28,28);
figure(1);
for i=1:10
    subplot(1,10,i)
    imshow(reshape(images_1(i,:,:),28,28));
    %colormap(spring);
end
%% get the feature and sample size of the data
[sample_size, feature_size]=size(training_set);
n_classes=10;

%% Let's start classifing
%In the Naive Bayes method,it supposes every feature has i.i.d
%distribution. so we can compute every dimensional feature using the
%statistical computing to caculate every feature's likelihood function
%---------------------------------------------------------------------%
%% Build the Bayesian Model 
priors = zeros(n_classes, 1);   
evidence = zeros(feature_size, 1);                                         %feature_size=size(training_set,2)
likelihoods = zeros(n_classes, feature_size);
%% Training:jump to the last paragraph code .

% for class=1:n_classes
%     fm=training_set(find(training_labels==class-1),:);
%     likelihoods(class,:)=sum(fm)./ size(fm,1);
%     priors(class)=size(fm,1)/sample_size;
% end
% %% test the probability
% if sum(priors)==1
%     disp('success');
% end
%% evidence:we have two method to get the evidence P(x)
evidence_1=(sum(training_set,1))./sample_size;
evidence_2=priors'*likelihoods;
%%compare the error :
% error=evidence_1-evidence_2;
% error_matrix=error(1,find(error(1,:)~=0));
% accuracy=1-size(error_matrix,2)/size(error,2)

%% traning and predict the new data's labels
 accuracy = 0.0;
    k = 0.0;
    %k_values=[0];
    k_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 60];%looking for the best k to smooth the probability
    for i=1:length(k_values)
        % train with k
        [c_likelihoods, c_priors, c_evidence] = naive_bayes_train(training_set, training_labels, n_classes, k_values(i));
        % classify cross-val set
        [crossval_predicted_classes, crossval_posteriors] = naive_bayes_classify(crossval_set, c_priors, c_likelihoods, c_evidence);
        % check if k is better
        c_accuracy = sum(crossval_predicted_classes == crossval_labels)/length(crossval_labels)*100.0;
        if c_accuracy>accuracy
            accuracy = c_accuracy;
            k = k_values(i);
            likelihoods = c_likelihoods;
            priors = c_priors;
            evidence = c_evidence;
        end
    end
    fprintf("Selected k=%2.2f with cross-validation accuracy=%2.2f%%.\n",k, accuracy);
    


    % trainingset accuray
    [trainingset_predicted_classes, trainingset_posteriors] = naive_bayes_classify(training_set, priors, likelihoods, evidence);
    accuracy = sum(trainingset_predicted_classes == training_labels)/length(training_labels)*100.0;
    fprintf("Accuracy on training-set=%2.2f%%\n", accuracy);

    % test
    [test_predicted_classes, test_posteriors] = naive_bayes_classify(test_set, priors, likelihoods, evidence);
    accuracy = sum(test_predicted_classes == test_labels)/length(test_labels)*100.0;
    fprintf("Accuracy on test-set=%2.2f%%\n", accuracy);
    %% save the training model
    saldir='./Training results/';
    save([saldir ,'likelihoods'],'likelihoods'); 
    save([saldir ,'priors'],'priors'); 
    save([saldir ,'evidence'],'evidence');