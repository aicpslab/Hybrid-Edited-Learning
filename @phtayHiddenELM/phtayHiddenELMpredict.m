function output=phtayHiddenELMpredict(phtayelm,input)
% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Example

% Yejiang Yang, 11-13-2024
% Copyright Yejiang Yang
%% Subfunction Generating the Hidden Layer
function MXR=PhyInput(X,r)
        MXR = [];
N=size(X,1);
Q=size(X,2);
% b. Generate Hidden Nodes
for l = 1:Q
    mxr = X(:, l);
    tempmxr = X(:, l);
    mcounting = (1:N)';
    for m = 2:r
        tb_list = [];
        for z = 1:N
            ta = X(z, l) * tempmxr(mcounting(z):end);
            if z == 1
                tb = ta;
            else
                tb = [tb; ta];
            end
            mcounting(z) = size(tb, 1) - length(ta) + 1;
        end
        tempmxr = tb;
        mxr = [mxr; tb];
    end

    if isempty(MXR)
        MXR = mxr;
    else
        MXR = [MXR, mxr];
    end
end
end

%% Main Function 
tic
%1. Input(R*Q) and Output(S*Q) 

P=input;
[~,Q]=size(P);
Structure=phtayelm.structure;
WeightGate=phtayelm.weightgate;

% The key difference lays how to generated randomized 
% 1.Passes Through Layer 
%Hidden = input;
TayLay=PhyInput(input,Structure(2,1));
switch phtayelm.activFcn
        case 'sig'
            TayLay = 1 ./ (1 + exp(-TayLay));
        case 'sin'
            TayLay = sin(TayLay);
        case 'hardlim'
            TayLay = hardlim(TayLay);
        case 'tansig'
            TayLay = tansig(TayLay);
        case 'ReLu'
            TayLay = max(TayLay,0);
        case 'purelin'
            TayLay = TayLay;
end
Hidden = [];
count=1;
for i = 1:size(WeightGate,2)
   if WeightGate(1,i)~=0
       Hidden(count,:)=TayLay(i,:);
       count=count+1;
   end
end
Hidden= phtayelm.weight{1}*Hidden;%+repmat(phtayelm.bias{2},1,Q);
output=Hidden;
end