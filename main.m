% Main program
program = classifyWines;
wine = program.ReadData('training_2014.csv');
[wine, training_raw, validation_raw] = program.ShuffleWines(wine, 0.7);

% Used arguments to classification with svm classifier
arguments = {
    'fixedAcidity','volatileAcidity','citricAcid', ...
    'residualSugar','chlorides','freeSulfurDioxide', ...
    'totalSulfurDioxide','density','pH','sulphates','alcohol' ...
};


%% K-nearest method to classify wines
modelWineType = program.WineTypeTraining(training_raw, arguments, 'type', 'k-nearest');
[errors1, prediction, scores] = program.WineTypeValidation(modelWineType, validation_raw, arguments, 'type', 'k-nearest');
% error1 ~ 6%

%% SVM method to classify wines
modelWineType = program.WineTypeTraining(training_raw, arguments, 'type', 'svm');
[errors2, prediction, scores] = program.WineTypeValidation(modelWineType, validation_raw, arguments, 'type', 'svm');
% error2 ~ 1%


%% K-nearest method to classify the quality of wines
model3 = program.WineTypeTraining(training_raw, arguments, 'quality', 'k-nearest');
[errors3, prediction, scores] = program.WineTypeValidation(model3, validation_raw, arguments, 'quality', 'k-nearest');
% base result: errors3 ~58%

%% Neural networks: classify the quality of wines
all_data = [training_raw; validation_raw];

net = patternnet(15);
%view(net)
net.performParam.ratio=0.5; 
net.trainParam.goal=1e-8;

x = table2array(training_raw(:,1:11));
y = table2array(training_raw(:,12));

t = zeros(size(x,1),7);
for i=1:7
   t(find(y==i), i) = 1; 
end

[net,tr] = train(net,x',t');
nntraintool
% error ~45%


%% Test model agaisnt TestData!
test_wine = program.ReadData('test_2014.csv');
[test_wine, test_raw, ~] = program.ShuffleWines(test_wine, 1.0);
[test_error, test_pred, test_score] = program.WineTypeValidation(SVMmodelWineType, test_raw, arguments, 'type');
