function elm=GeneratephtayELM(Structure,Phy_lay,activation)
% GeneratePhysicsRegulatedELM Create a Physics Regulated Extreme Learning Machine
% Syntax
% elm = =GeneratePhysicsRegulatedELM(R,N,activeFcn,S)
% Description
% Structure  - Stucture of PhysicsRegulatedELM(Layers*[TaylorRank,OutputNeuronNum]))
% activeFcn  - Transfer Function:
          % 'sig' for Sigmoidal function (default) 
          % 'sin' for Sine function
          % 'hardlim' for Hardlim function
% Output
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)


% Phy_lay_B (Oup_Dim*Inp_Dim) indicates the weather the Input and Output is related(1) or not(0)
% Yejiang Yang,11-12-2024
% Copyright Yejiang Yang

%% Inner Relation Function
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

Phy_lay_B_save=Phy_lay;
A=zeros(size(Phy_lay,1),1);
for i = 1:size(Phy_lay,1)
    if(min(Phy_lay(i,:))==0)
        A(i,1)=1;
    end
end
%% The key difference lays how to generated randomized 
% 1.GenerateWeight
elm.stucture=Structure;
HiddenGate=Phy_lay;
%for i = 1:size(Structure,1)

for i = 1:1
InN=size(PhyInput(ones(Structure(1,1),1),Structure(1,2)),1);
    %HiddenGate{i}=ones(Structure(i,2),InN);
   % for j = 1:size(Phy_lay,1)
    %  HiddenGate{i}(j,:)=PhyInput(Phy_lay(j,:)',Structure(i,1))';
    %end
    %weight{i}=1e-3*rand(Structure(i,2),InN).*HiddenGate{i};
   % weight{i}=rand(Structure(i,2),InN).*repmat(HiddenGate{i},1,2);
   weight{i}=rand(InN,Structure(1,1));
end
weight{2}=repmat(HiddenGate,1,2);
elm.bias{1}=Phy_lay;

% Randomly Generate the Input Weight Matrix
    % elm.weight{1} = rand(N,R) * 2 - 1;
    % elm.weight{2} = rand(S,N) * 2;
    % Randomly Generate the Bias Matrix
    % elm.bias{1} = rand(N,1);
    % elm.bias{2} = zeros(S,1);
    %for i = 1:size(Structure,1)-1
    % for i = 1:1
    %     elm.bias{i}=rand(Structure(i,2),1);
    % end
 %   elm.bias{size(Structure,1)}=zeros(Structure(end,2),1);
    
    elm.trainingError = [];
    elm=phtayELM(Structure,Phy_lay,weight,elm.bias,activation);
end
