function phtayelm = trainphtayELMcons(phtayelm,Input,Output)
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
Phy_lay=phtayelm.phy_lay;

% The key difference lays how to generated randomized 
% 1.Passes Through Layer 
Hidden = Input;
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
    Hidden = phtayelm.weight{i}*TayLay+repmat(phtayelm.bias{i},1,Q);
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

 LW=zeros(size(phtayelm.weight{2}));
for j = 1:size(phtayelm.weight{2},1)

 if(min(abs(phtayelm.weight{2}(j,:)))==0)
    for counter = 1:size(phtayelm.weight{2}(j,:),2)
        if phtayelm.weight{2}(j,counter)
           answer=(pinv(Hidden(counter,:)') * T(j,:)')';
           LW(j,counter)=answer;
        end
    end
 else
    
   % LW(j,:)=(pinv(Hidden') * T(j,:)')';
   A= [Hidden';-Hidden'];
   b=[repmat([phtayelm.cons(j,1)],Q,1);repmat([phtayelm.cons(j,2)],Q,1)];
   % size(A)
   % size(b)
   LW(j,:)=lsqlin(Hidden', T(j,:)',A, b);
end
end
% Calculate the Output Weight Matrix
phtayelm.weight{2}=LW;
phtayelm=phtayELM(Structure,Phy_lay,phtayelm.weight,phtayelm.bias,phtayelm.activFcn);

% fprintf('time of training ELM is %d', toc)
% fprintf(' seconds \n')

%% Traing Error
phtayelm.trainingError=mse(Output-phtayELMpredict(phtayelm,Input));
%elm.trainingError = partitions.MeanSquare(X);

end