function output=phtayELMpredictcons(phtayelm,input)
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
Phy_lay=phtayelm.phy_lay;

% The key difference lays how to generated randomized 
% 1.Passes Through Layer 
Hidden = input;
%for i = 1:size(Structure,1)
for i = 1:1
    % InN=size(PhyInput(Phy_lay(1,:)',Structure(i,1)),1);
    % HiddenGate{i}=ones(Structure(i,2),InN);
    % for j = 1:size(Phy_lay,1)
    %   HiddenGate{i}(j,:)=PhyInput(Phy_lay(j,:)',Structure(i,1))';
    % end
    % weight{i}=rand(Structure(i,2),InN).*HiddenGate{i};
    % Phy_lay=[];
    % Phy_lay=ones(size(Phy_lay_B_save,1),Structure(i,2));
    % for k = 1:size(Phy_lay_B_save,1)
    %     if A(k,1)==1
    %         Phy_lay(k,:)=0;
    %         Phy_lay(k,k)=1;
    %     end
    % end
    TayLay=PhyInput(Hidden,Structure(i,1));
    Hidden= phtayelm.weight{i}*TayLay+repmat(phtayelm.bias{i},1,Q);
    switch phtayelm.activFcn
        case 'sig'
            Hidden = 1 ./ (1 + exp(-Hidden));
        case 'sin'
            Hidden = sin(Hidden);
        case 'hardlim'
            Hidden = hardlim(Hidden);
        case 'tansig'
            Hidden = tansig(Hidden);
        case 'ReLu'
            Hidden = max(Hidden,0);
        case 'purelin'
            Hidden = Hidden;
    end
end
Hidden= phtayelm.weight{2}*Hidden+repmat(phtayelm.bias{2},1,Q);
output=Hidden;
end