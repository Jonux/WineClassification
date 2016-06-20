
function program = classifyWines
  program.ReadData = @ReadData;
  program.ShuffleWines = @ShuffleWines;
  
  program.WineTypeTraining=@WineTypeTraining;
  program.WineTypeValidation=@WineTypeValidation;
  program.WineTypeTesting=@WineTypeTesting;
end

function wine = ReadData(file)
    wine = readtable(file);
end

% shuffle wines and divide the set to training and validation
function [wine, training_raw, validation_raw] = ShuffleWines(wine, trainingSize)
    wine = wine(randperm(height(wine)),:);
    %reds = wine(strcmp(wine.type, 'Red'), :);
    %whites = wine(strcmp(wine.type, 'White'), :);
    
    n_training = round(height(wine)*trainingSize);
    
    training_raw = wine(1:n_training, :);
    validation_raw = wine(n_training:end, :);
end

function [models preds] = multiClassSVM(X, Y, numLabels)
    models = cell(numLabels,1);
    preds = {}
    for k=1:numLabels
        binaryResults = zeros(size(X,1),1);
        for a=1:size(X,1)
           if Y(a) == k
               binaryResults(a) = 1;
           end
        end
        models{k} = fitcsvm(Y, binaryResults);
        preds = predict(mdl{k}, arrayXw(i,k));
    end
end

% returns model of data
function [SVMmodel models] = WineTypeTraining(raw_data, used_arguments, classifierArgument, classifierMethod)
    Xw = raw_data(:,used_arguments); % use all data for fitting
    Yw = raw_data(:,classifierArgument); % response data type
    X = table2array(Xw);
    models = 0;
    
    if strcmp(classifierArgument, 'type')
        Y = ones(size(Yw(:,1),1),1); 
        indicesY = find(cellfun(@(x) strcmp(x,'Red'), raw_data.type));
        Y(indicesY) = 0;
        % now red = 0, white = 1
        if strcmp(classifierMethod, 'k-nearest')
            SVMmodel = fitcknn(X,Y)
        else
            SVMmodel = fitcsvm(X,Y)%,'KernelFunction','mysigmoid','Standardize', true)
        end
        disp('type');
    elseif strcmp(classifierArgument, 'quality')
        Y = raw_data.quality;
        if strcmp(classifierMethod, 'k-nearest')
            SVMmodel = fitcknn(X,Y) % k-nearest neighbours
        
            disp('k-nearest');
        else % multi SVM modeling
             disp('Testing');
            Y2 = Y;
            numLabels = length(unique(Y2));
            models = multiClassSVM(X,Y2,numLabels);
            SVMmodel = 0;
        end
    else
        disp('ERROR: arguments');
    end
end

function [error, prediction, scores] = WineTypeValidation(mdl, raw_data, used_arguments, classifierArgument, classifierMethod)
    validationXw = raw_data(:,used_arguments);

    if strcmp(classifierArgument, 'type')
        validationYw = raw_data(:,classifierArgument); % response data
        validationY = ones(size(validationYw(:,1),1),1); 
        indices = find(cellfun(@(x) strcmp(x,'Red'), raw_data.type));
        validationY(indices) = 0;
        [prediction,scores] = predict(mdl, table2array(validationXw));
        
    elseif strcmp(classifierArgument, 'quality')
        validationY = raw_data.quality;
        
        if strcmp(classifierMethod, 'k-nearest')
            [prediction,scores] = predict(mdl, table2array(validationXw));
            disp('WineTypeValidation: k-nearest');
        else 
            % mdl is array of classifiers
            Y2 = Y;
            numLabels = length(unique(Y2));
            models = multiClassSVM(X,Y2,numLabels);
            SVMmodel = 0;

%             prediction = zeros(size(validationXw,1),1);
%             arrayXw = table2array(validationXw);
%             for i=1:size(validationXw,1)
%                 for k:7
%                    [preds,scores] = predict(mdl{k}, arrayXw(i,k));
%                 end
%             end
        end
    else
        disp('ERROR: arguments');
    end
    A = [prediction validationY];
    error = sum(prediction~=validationY)/size(raw_data,1)
end

function z=WineTypeTestAndRate
  z=1;
end
