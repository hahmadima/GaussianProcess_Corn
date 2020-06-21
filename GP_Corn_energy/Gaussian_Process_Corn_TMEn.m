clear all; clc;

%% Data 
run Data_bz; close all;
dl=info(:,1);
inp=info(:,2:5);
out=info(:,6);

% Train a GP regression model
% This code specifies all the model options and trains the model.
GPRm = fitrgp(...
    inp, ...
    out, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'squaredexponential', ...  
    'Standardize', true);
% 'squaredexponential' 'matern52' 'matern32'  'ardsquaredexponential'
    % 'ardmatern32' 'ardmatern52';

ypred=predict(GPRm, inp)


% Perform cross-validation
partitionedModel = crossval(GPRm, 'KFold', 5);
% Compute validation predictions
yval = kfoldPredict(partitionedModel);
% Compute goodness of fits

RMSEpred = sqrt(mean((ypred-out) .^ 2))
R2pred= 1 - RMSEpred^2 / var(out, 1)

RMSEval = sqrt(mean((yval-out) .^ 2))
R2val= 1 - RMSEval^2 / var(out, 1)
res_pred=ypred-out;
res_val=yval-out;


%% Plotting results
% Tarining
subplot(231);plot(dl,out,'.',dl,ypred,'-');legend('Actual','Predicted');xlabel('Data lines');ylabel('Y');
title(['Training' '   R^2= ' num2str(R2pred) '  RMSE = ' num2str(RMSEpred)]);grid on;
% Validation
subplot(234);plot(dl,out,'.',dl,yval,'-');legend('Actual','Validation');xlabel('Data lines');ylabel('Y');
title(['Validating' '   R^2= ' num2str(R2val) '  RMSE = ' num2str(RMSEval)]);grid on;
% Actual vs predicted
subplot(232);scatter(out,ypred,'k.');;xlabel('Actual');ylabel('Predicted');
title(['Actual vs predicted' '   R^2= ' num2str(R2pred) '  RMSE = ' num2str(RMSEpred)]);grid on;
% Actual vs validation
subplot(235);scatter(out,yval,'k.');;xlabel('Actual');ylabel('Validation');
title(['Actual vs validation' '   R^2= ' num2str(R2val) '  RMSE = ' num2str(RMSEval)]);grid on;
% Predicted vs residual
subplot(233);scatter(ypred,res_pred,'k.');;xlabel('Predicted');ylabel('Residual');
title(['Predicted vs residual' '   R^2= ' num2str(R2pred) '  RMSE = ' num2str(RMSEpred)]);grid on;
% Vaidation vs residual
subplot(236);scatter(yval,res_val,'k.');;xlabel('Validation');ylabel('Residual');
title(['Validation vs residual' '   R^2= ' num2str(R2val) '  RMSE = ' num2str(RMSEval)]);grid on;

% subplot(221);createfigure(out, ypred);
% subplot(222);createfigure(out, yval);
% subplot(223);createfigure(ypred, res_pred);
% subplot(224);createfigure(yval, res_val);

createfigure(out, ypred);
createfigure(out, yval);
createfigure(ypred, res_pred);
createfigure(yval, res_val);


OUT=[inp out ypred yval res_pred res_val];