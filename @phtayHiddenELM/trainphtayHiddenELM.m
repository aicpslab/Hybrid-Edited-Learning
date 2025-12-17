function phtayelm = trainphtayHiddenELM(phtayelm,Input,Output)
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax
% [IW,B,LW,TF,TYPE] = elmtrain(phytayELM,Input,Output)
% Description
% Input  - Input Matrix of Training Set  (R*Q)
% Output  - Output Matrix of Training Set (S*Q)
% TYPE - Regression (0,default) or Classification (1)
% Output
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

P=Input;
[~,Q]=size(P);
T=Output;
Structure=phtayelm.structure;
WeightGate=phtayelm.weightgate;

% The key difference lays how to generated randomized 
% 1.Passes Through Layer 

%for i = 1:size(Structure,1)
% for i = 1:1

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
    
    % Hidden = phtayelm.weight{i}*TayLay+repmat(phtayelm.bias{i},1,Q);
%     switch phtayelm.activFcn
%         case 'sig'
%             Hidden = 1 ./ (1 + exp(-Hidden));
%         case 'sin'
%             Hidden = sin(Hidden);
%         case 'hardlim'
%             Hidden = hardlim(Hidden);
%         case 'tansig'
%             Hidden = tansig(Hidden);
%         case 'ReLu'
%             Hidden = max(Hidden,0);
%         case 'purelin'
%             Hidden = Hidden;
%     end
% end

TayLay=PhyInput(Input,Structure(2,1));
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
 %LW=zeros(size(phtayelm.weight{1}));

answer=(pinv(Hidden') * T')';
           LW=answer;% Calculate the Output Weight Matrix
phtayelm.weight{1}=LW;
phtayelm=phtayHiddenELM(Structure,WeightGate,phtayelm.weight,phtayelm.bias,phtayelm.activFcn);

% fprintf('time of training ELM is %d', toc)
% fprintf(' seconds \n')

%% Traing Error
phtayelm.trainingError=mse(Output,phtayHiddenELMpredict(phtayelm,Input));
%elm.trainingError = partitions.MeanSquare(X);

end